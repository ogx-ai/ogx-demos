"""
Demo: Multi-Source RAG Crawler

Description:
This demo teaches how to build a cross-source document crawler that follows
links between different platforms (GitHub, GitLab, Confluence, Jira, Google Docs)
and indexes all discovered documents into a Llama Stack vector store for RAG
queries.

Most RAG examples index documents from a single source. In practice, knowledge
is spread across platforms -- a GitHub README links to a design doc, which links
to an issue tracker, which references another repo. This demo shows how to
automatically discover and index that connected knowledge graph.

Supported source types:
- GitHub repositories and files (public or via GITHUB_TOKEN)
- GitLab repositories and files (via GITLAB_TOKEN)
- Confluence pages (via CONFLUENCE_URL + CONFLUENCE_TOKEN)
- Jira issues (via JIRA_URL + JIRA_TOKEN)
- Google Docs (via GOOGLE_API_KEY or service account)
- Generic web pages (markdown, text, HTML)

Learning Objectives:
- Detect and extract links to different platforms from document content
- Crawl documents breadth-first with configurable depth limits
- Track crawl provenance (which document linked to which)
- Index crawled documents into Llama Stack vector stores with metadata
- Query across all discovered documents with file_search
- Use incremental indexing with content hashing to avoid re-processing
- Add new source types by registering a fetcher function
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import hashlib
import html
import os
import re
import socket
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from ipaddress import ip_address
from typing import Callable
from uuid import uuid4

import fire
from llama_stack_client import LlamaStackClient
from termcolor import colored

from demos.shared.utils import (
    can_model_chat,
    check_model_is_available,
    get_any_available_chat_model,
    get_any_available_embedding_model,
    get_embedding_dimension,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    import requests as _requests
except ImportError:  # pragma: no cover - optional dependency
    _requests = None

# Maximum bytes to download from any single URL.
MAX_DOWNLOAD_BYTES = 15_000


def _maybe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _strip_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _redact_url(url: str) -> str:
    """Strip query string and fragment from a URL to avoid leaking secrets."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _is_safe_url(url: str) -> bool:
    """Reject URLs that resolve to private/loopback/reserved addresses (SSRF guard)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        for _family, _socktype, _proto, _canonname, sockaddr in socket.getaddrinfo(
            parsed.hostname, None
        ):
            addr = ip_address(sockaddr[0])
            if (
                addr.is_private
                or addr.is_loopback
                or addr.is_link_local
                or addr.is_reserved
                or addr.is_multicast
            ):
                return False
    except (socket.gaierror, ValueError):
        return False
    return True


def _safe_download(url: str, headers: dict | None = None) -> str | None:
    """Download text content from a URL with size and content-type guards.

    Returns the text content (up to MAX_DOWNLOAD_BYTES) or None on failure.
    """
    if _requests is None:
        return None

    resp = _requests.get(
        url, headers=headers or {}, timeout=(5, 15), stream=True, allow_redirects=False
    )
    if resp.status_code != 200:
        return None

    content_type = resp.headers.get("Content-Type", "").lower()
    if content_type and "text" not in content_type and "json" not in content_type:
        resp.close()
        return None

    content_length = int(resp.headers.get("Content-Length", "0") or 0)
    if content_length > MAX_DOWNLOAD_BYTES * 10:
        resp.close()
        return None

    raw_bytes = resp.raw.read(MAX_DOWNLOAD_BYTES, decode_content=True)
    resp.close()
    return raw_bytes.decode(resp.encoding or "utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Link Detection
# ---------------------------------------------------------------------------


class LinkType(str, Enum):
    """Supported link/source types for crawling."""

    GITHUB = "github"
    GITLAB = "gitlab"
    CONFLUENCE = "confluence"
    JIRA = "jira"
    GOOGLE_DOCS = "google_docs"
    WEB = "web"


@dataclass
class DetectedLink:
    """A link detected in document content."""

    link_type: LinkType
    identifier: str  # Source-specific ID (e.g. "owner/repo:path" for GitHub)
    url: str  # Original URL that matched


# Regex patterns for detecting links to different platforms.
# The GitHub pattern only matches blob/tree content paths or bare repo roots
# (terminated by whitespace/punctuation), avoiding false matches on /issues/,
# /pull/, /releases/ etc.
LINK_PATTERNS: dict[LinkType, re.Pattern] = {
    LinkType.GITHUB: re.compile(
        r"https?://github\.com/([^/\s]+/[^/\s]+)"
        r"(?:/(?:blob|tree)/[^/\s]+/([^\s)\"<>]*))?"
        r"(?=$|[\s)\"<>])",
        re.IGNORECASE,
    ),
    LinkType.GITLAB: re.compile(
        r"https?://gitlab[^/\s]*/([^/\s]+/[^/\s]+)"
        r"(?:/[-/]?(?:blob|tree)/[^/\s]+/([^\s)\"<>]*))?"
        r"(?=$|[\s)\"<>])",
        re.IGNORECASE,
    ),
    LinkType.CONFLUENCE: re.compile(
        r"https?://[^\s/]*confluence[^\s/]*(?:/wiki)?/"
        r"(?:spaces/([^/\s]+)/pages/(\d+)|pages/viewpage\.action\?pageId=(\d+))",
        re.IGNORECASE,
    ),
    LinkType.JIRA: re.compile(
        r"https?://[^\s/]*(?:jira|issues)[^\s/]*/browse/([A-Z]+-\d+)",
        re.IGNORECASE,
    ),
    LinkType.GOOGLE_DOCS: re.compile(
        r"https?://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)",
        re.IGNORECASE,
    ),
    LinkType.WEB: re.compile(
        r"https?://[^\s)\"<>]+\.(?:md|txt|rst|html?)(?:\?[^\s)\"<>]*)?",
        re.IGNORECASE,
    ),
}


def detect_links(
    text: str, enabled_types: list[LinkType] | None = None
) -> list[DetectedLink]:
    """Scan text for links to supported platforms.

    Args:
        text: Content to scan for links.
        enabled_types: Limit detection to these types. ``None`` means all.

    Returns:
        Deduplicated list of detected links.
    """
    types = enabled_types or list(LinkType)
    seen: set[str] = set()
    links: list[DetectedLink] = []

    for link_type in types:
        pattern = LINK_PATTERNS.get(link_type)
        if pattern is None:
            continue

        for match in pattern.finditer(text):
            url = match.group(0)
            if url in seen:
                continue
            seen.add(url)

            identifier = _extract_identifier(link_type, match, url)
            links.append(DetectedLink(link_type=link_type, identifier=identifier, url=url))

    return links


def _extract_identifier(link_type: LinkType, match: re.Match, url: str) -> str:
    """Extract a source-specific identifier from a regex match."""
    if link_type == LinkType.GITHUB:
        repo = match.group(1)
        path = match.group(2)
        return f"{repo}:{path}" if path else repo

    if link_type == LinkType.GITLAB:
        repo = match.group(1)
        path = match.group(2)
        return f"{repo}:{path}" if path else repo

    if link_type == LinkType.CONFLUENCE:
        space, page_id, viewpage_id = match.group(1), match.group(2), match.group(3)
        if page_id:
            return f"{space}:{page_id}"
        if viewpage_id:
            return viewpage_id
        return url

    if link_type == LinkType.JIRA:
        return match.group(1)

    if link_type == LinkType.GOOGLE_DOCS:
        return match.group(1)

    return url


# ---------------------------------------------------------------------------
# Crawl Models
# ---------------------------------------------------------------------------


@dataclass
class CrawlTrace:
    """Provenance information: how a document was discovered."""

    seed_url: str
    parent_url: str | None = None
    parent_title: str | None = None
    depth: int = 0
    path: list[str] = field(default_factory=list)

    def child(self, parent_url: str, parent_title: str | None = None) -> CrawlTrace:
        return CrawlTrace(
            seed_url=self.seed_url,
            parent_url=parent_url,
            parent_title=parent_title,
            depth=self.depth + 1,
            path=[*self.path, parent_url],
        )


@dataclass
class CrawledDocument:
    """A document fetched by the crawler."""

    url: str
    title: str
    content: str
    source_type: LinkType
    trace: CrawlTrace
    metadata: dict = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return self.trace.depth

    @property
    def content_hash(self) -> str:
        """SHA-256 prefix for change detection."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


@dataclass
class CrawlStats:
    """Aggregate statistics from a crawl run."""

    total_fetched: int = 0
    total_skipped: int = 0
    total_failed: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_depth: dict[int, int] = field(default_factory=dict)

    def record(self, doc: CrawledDocument) -> None:
        self.total_fetched += 1
        src = doc.source_type.value
        self.by_source[src] = self.by_source.get(src, 0) + 1
        self.by_depth[doc.depth] = self.by_depth.get(doc.depth, 0) + 1


# ---------------------------------------------------------------------------
# Crawler Service
# ---------------------------------------------------------------------------


@dataclass
class _QueueItem:
    link_type: LinkType
    identifier: str
    url: str
    trace: CrawlTrace


class MultiSourceCrawler:
    """Breadth-first crawler that follows links across platforms.

    Register fetcher functions for each :class:`LinkType`, then call
    :meth:`crawl` to discover and collect documents.

    Example::

        crawler = MultiSourceCrawler(max_depth=2)
        crawler.register_fetcher(LinkType.GITHUB, fetch_github)
        crawler.add_seed("https://github.com/org/repo", LinkType.GITHUB, "org/repo")
        documents = crawler.crawl()
    """

    def __init__(
        self,
        max_depth: int = 2,
        max_docs: int = 50,
    ) -> None:
        self.max_depth = max_depth
        self.max_docs = max_docs
        self._queue: deque[_QueueItem] = deque()
        self._visited: set[str] = set()
        self._documents: list[CrawledDocument] = []
        self._stats = CrawlStats()
        self._fetchers: dict[LinkType, Callable[[str, str, CrawlTrace], CrawledDocument | None]] = {}
        self._seen_hashes: set[str] = set()

    def register_fetcher(
        self,
        link_type: LinkType,
        fetcher: Callable[[str, str, CrawlTrace], CrawledDocument | None],
    ) -> None:
        """Register a document fetcher for a link type.

        The fetcher signature is ``(identifier, url, trace) -> CrawledDocument | None``.
        """
        self._fetchers[link_type] = fetcher

    def add_seed(self, url: str, link_type: LinkType, identifier: str) -> None:
        """Add a seed URL to start crawling from."""
        trace = CrawlTrace(seed_url=url, depth=0)
        self._queue.append(_QueueItem(link_type=link_type, identifier=identifier, url=url, trace=trace))

    def crawl(self) -> list[CrawledDocument]:
        """Run the crawl. Returns all fetched documents."""
        while self._queue:
            if self._stats.total_fetched >= self.max_docs:
                print(colored(f"  Reached max documents limit ({self.max_docs})", "yellow"))
                break

            item = self._queue.popleft()

            if item.url in self._visited:
                continue
            if item.trace.depth > self.max_depth:
                self._stats.total_skipped += 1
                continue

            self._visited.add(item.url)

            fetcher = self._fetchers.get(item.link_type)
            if fetcher is None:
                continue

            try:
                doc = fetcher(item.identifier, item.url, item.trace)
            except Exception as exc:
                print(colored(f"  Failed to fetch {item.url}: {exc}", "red"))
                self._stats.total_failed += 1
                continue

            if doc is None:
                self._stats.total_failed += 1
                continue

            # Deduplicate by content hash (same content at different URLs)
            if doc.content_hash in self._seen_hashes:
                self._stats.total_skipped += 1
                continue
            self._seen_hashes.add(doc.content_hash)

            self._stats.record(doc)
            self._documents.append(doc)

            depth_indicator = "  " * doc.depth
            print(f"  {depth_indicator}[depth={doc.depth}] {doc.source_type.value}: {doc.title}")

            # Discover and queue child links
            if doc.depth < self.max_depth:
                child_trace = doc.trace.child(doc.url, doc.title)
                for link in detect_links(doc.content):
                    if link.url not in self._visited and _is_safe_url(link.url):
                        self._queue.append(
                            _QueueItem(
                                link_type=link.link_type,
                                identifier=link.identifier,
                                url=link.url,
                                trace=child_trace,
                            )
                        )

        return self._documents

    @property
    def stats(self) -> CrawlStats:
        return self._stats


# ---------------------------------------------------------------------------
# Built-in Fetchers
# ---------------------------------------------------------------------------


def fetch_github(
    identifier: str, url: str, trace: CrawlTrace
) -> CrawledDocument | None:
    """Fetch a file from GitHub via the raw content URL.

    Works for public repositories without authentication.
    Set ``GITHUB_TOKEN`` env var to access private repos.
    """
    # Validate hostname to prevent token leakage to lookalike domains (CWE-200).
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname != "github.com":
        return None

    # Build raw URL from github.com URL
    raw_url = (
        url.replace("github.com", "raw.githubusercontent.com")
        .replace("/blob/", "/")
        .replace("/tree/", "/")
    )
    if raw_url == url:
        # URL was just a repo root -- try fetching README
        repo = identifier.split(":")[0]
        raw_url = f"https://raw.githubusercontent.com/{repo}/main/README.md"

    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    # Only send token to the real raw.githubusercontent.com host.
    raw_parsed = urllib.parse.urlparse(raw_url)
    if token and raw_parsed.hostname == "raw.githubusercontent.com":
        headers["Authorization"] = f"token {token}"

    content = _safe_download(raw_url, headers)
    if content is None:
        return None

    repo = identifier.split(":")[0]
    path = identifier.split(":", 1)[1] if ":" in identifier else "README.md"

    return CrawledDocument(
        url=url,
        title=f"{repo}/{path}",
        content=content,
        source_type=LinkType.GITHUB,
        trace=trace,
        metadata={"repo": repo, "path": path},
    )


def fetch_gitlab(
    identifier: str, url: str, trace: CrawlTrace
) -> CrawledDocument | None:
    """Fetch a file from a GitLab repository.

    Requires ``GITLAB_TOKEN`` env var. Optionally set ``GITLAB_URL`` to override
    the base URL (defaults to extracting it from the link URL).

    Example env::

        GITLAB_URL=https://gitlab.com
        GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx
    """
    if _requests is None:
        return None

    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        return None

    # Determine GitLab base URL from the matched URL
    parsed = urllib.parse.urlparse(url)
    base_url = os.environ.get("GITLAB_URL", f"{parsed.scheme}://{parsed.netloc}")

    repo = identifier.split(":")[0]
    path = identifier.split(":", 1)[1] if ":" in identifier else "README.md"
    project_encoded = urllib.parse.quote(repo, safe="")

    # Try main branch, then master
    for ref in ("main", "master"):
        file_url = (
            f"{base_url}/api/v4/projects/{project_encoded}"
            f"/repository/files/{urllib.parse.quote(path, safe='')}/raw"
        )
        headers = {"PRIVATE-TOKEN": token}
        content = _safe_download(f"{file_url}?ref={ref}", headers)
        if content is not None:
            web_url = f"{base_url}/{repo}/-/blob/{ref}/{path}"
            return CrawledDocument(
                url=web_url,
                title=f"{repo}/{path}",
                content=content,
                source_type=LinkType.GITLAB,
                trace=trace,
                metadata={"repo": repo, "path": path, "ref": ref},
            )

    return None


def fetch_confluence(
    identifier: str, url: str, trace: CrawlTrace
) -> CrawledDocument | None:
    """Fetch a Confluence page by page ID.

    Requires ``CONFLUENCE_URL`` and ``CONFLUENCE_TOKEN`` env vars.

    Example env::

        CONFLUENCE_URL=https://confluence.example.com
        CONFLUENCE_TOKEN=your-personal-access-token
    """
    if _requests is None:
        return None

    confluence_url = os.environ.get("CONFLUENCE_URL")
    token = os.environ.get("CONFLUENCE_TOKEN")
    if not confluence_url or not token:
        return None

    # Extract page ID from identifier ("SPACE:12345" or just "12345")
    page_id = identifier.split(":")[-1] if ":" in identifier else identifier

    api_url = f"{confluence_url}/rest/api/content/{page_id}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"expand": "body.storage,space"}

    resp = _requests.get(api_url, headers=headers, params=params, timeout=(5, 15))
    if resp.status_code != 200:
        return None

    data = resp.json()
    title = data.get("title", f"Page {page_id}")
    space_key = data.get("space", {}).get("key", "")
    body_html = data.get("body", {}).get("storage", {}).get("value", "")
    body_text = _strip_html(body_html)

    web_url = f"{confluence_url}/wiki/spaces/{space_key}/pages/{page_id}"

    content = f"Confluence Page: {title}\nSpace: {space_key}\nURL: {web_url}\n\n{body_text}"

    return CrawledDocument(
        url=web_url,
        title=title,
        content=content[:MAX_DOWNLOAD_BYTES],
        source_type=LinkType.CONFLUENCE,
        trace=trace,
        metadata={"page_id": page_id, "space": space_key},
    )


def fetch_jira(
    identifier: str, url: str, trace: CrawlTrace
) -> CrawledDocument | None:
    """Fetch a Jira issue by key (e.g. PROJECT-123).

    Requires ``JIRA_URL`` and ``JIRA_TOKEN`` env vars.

    Example env::

        JIRA_URL=https://jira.example.com
        JIRA_TOKEN=your-personal-access-token
    """
    if _requests is None:
        return None

    jira_url = os.environ.get("JIRA_URL")
    token = os.environ.get("JIRA_TOKEN")
    if not jira_url or not token:
        return None

    issue_key = identifier  # e.g. "PROJECT-123"
    api_url = f"{jira_url}/rest/api/2/issue/{issue_key}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"fields": "summary,description,status,project,issuetype,comment"}

    resp = _requests.get(api_url, headers=headers, params=params, timeout=(5, 15))
    if resp.status_code != 200:
        return None

    data = resp.json()
    fields = data.get("fields", {})
    summary = fields.get("summary", "")
    description = fields.get("description", "") or ""
    status = fields.get("status", {}).get("name", "")
    project = fields.get("project", {}).get("name", "")
    issue_type = fields.get("issuetype", {}).get("name", "")

    web_url = f"{jira_url}/browse/{issue_key}"

    content = (
        f"Jira Issue: {issue_key}\n"
        f"Summary: {summary}\n"
        f"Project: {project}\n"
        f"Type: {issue_type}\n"
        f"Status: {status}\n"
        f"URL: {web_url}\n\n"
        f"Description:\n{description[:5000]}"
    )

    # Include recent comments if present
    comments = fields.get("comment", {}).get("comments", [])
    if comments:
        content += "\n\nComments:\n"
        for comment in comments[-5:]:
            author = comment.get("author", {}).get("displayName", "Unknown")
            body = comment.get("body", "")[:500]
            content += f"\n[{author}]: {body}\n"

    return CrawledDocument(
        url=web_url,
        title=f"{issue_key}: {summary}",
        content=content[:MAX_DOWNLOAD_BYTES],
        source_type=LinkType.JIRA,
        trace=trace,
        metadata={"key": issue_key, "project": project, "status": status},
    )


def fetch_google_doc(
    identifier: str, url: str, trace: CrawlTrace
) -> CrawledDocument | None:
    """Fetch a Google Doc by document ID.

    Requires either ``GOOGLE_API_KEY`` env var (simplest) or a Google service
    account credentials file at ``GOOGLE_APPLICATION_CREDENTIALS``.

    Example env::

        GOOGLE_API_KEY=AIzaSy...
    """
    if _requests is None:
        return None

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    doc_id = identifier
    api_url = f"https://docs.googleapis.com/v1/documents/{doc_id}"
    params = {"key": api_key}

    resp = _requests.get(api_url, params=params, timeout=(5, 15))
    if resp.status_code != 200:
        return None

    data = resp.json()
    title = data.get("title", f"Doc {doc_id}")

    # Extract text from the document body
    text_parts: list[str] = []
    for element in data.get("body", {}).get("content", []):
        paragraph = element.get("paragraph", {})
        for text_element in paragraph.get("elements", []):
            text_run = text_element.get("textRun", {})
            text = text_run.get("content", "")
            if text.strip():
                text_parts.append(text)

    body_text = "".join(text_parts)
    web_url = f"https://docs.google.com/document/d/{doc_id}"

    content = f"Google Doc: {title}\nURL: {web_url}\n\n{body_text}"

    return CrawledDocument(
        url=web_url,
        title=title,
        content=content[:MAX_DOWNLOAD_BYTES],
        source_type=LinkType.GOOGLE_DOCS,
        trace=trace,
        metadata={"doc_id": doc_id},
    )


def fetch_web_page(
    identifier: str, url: str, trace: CrawlTrace
) -> CrawledDocument | None:
    """Fetch a plain-text or markdown web page."""
    if not _is_safe_url(url):
        return None

    content = _safe_download(url)
    if content is None:
        return None

    title = url.rsplit("/", 1)[-1] or url

    return CrawledDocument(
        url=url,
        title=title,
        content=content,
        source_type=LinkType.WEB,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:
                    chunk = chunk[: last_sep + len(sep)]
                    end = start + len(chunk)
                    break

        stripped = chunk.strip()
        if stripped:
            chunks.append(stripped)
        start = end - overlap

    return chunks


def _to_dict(value) -> dict | None:
    """Convert an SDK object to a dict if possible."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return None


def _print_retrieved_sources(response) -> None:
    """Extract and display file_search results from a Responses API response."""
    output_items = getattr(response, "output", None) or []
    if isinstance(output_items, dict):
        output_items = [output_items]
    if not isinstance(output_items, list):
        return

    for item in output_items:
        item_dict = _to_dict(item) or {}
        if item_dict.get("type") != "file_search_call":
            continue
        results = (
            item_dict.get("results")
            or item_dict.get("file_search_call", {}).get("results")
            or []
        )
        if not results:
            print(colored("[sources] file_search returned no results", "yellow"))
            return
        print(f"\n  Retrieved {len(results)} chunks:")
        for result in results[:5]:
            r = _to_dict(result) or {}
            filename = r.get("filename", "unknown")
            score = r.get("score", "n/a")
            content_parts = r.get("content") or []
            snippet = ""
            for chunk in content_parts:
                chunk_dict = _to_dict(chunk) or {}
                text = chunk_dict.get("text")
                if text:
                    snippet += text.strip() + " "
            snippet = snippet.strip()
            # Show the source URL if embedded in the chunk
            source_line = ""
            if snippet.startswith("Source:"):
                first_line = snippet.split("\n", 1)[0]
                source_line = f" | {first_line}"
            print(f"  - {filename} (score={score}){source_line}")
        return

    print(colored("[sources] no file_search results in response", "yellow"))


def index_crawled_documents(
    client: LlamaStackClient,
    vector_store_id: str,
    documents: list[CrawledDocument],
    chunk_size: int = 512,
) -> int:
    """Index crawled documents into a Llama Stack vector store.

    Each document is chunked and uploaded as a file. Metadata (source URL,
    crawl depth, provenance path) is embedded in the content so the LLM
    can cite sources. URLs are redacted (query strings stripped) to avoid
    persisting secrets like signed tokens.

    Returns the number of chunks indexed.
    """
    indexed = 0
    for doc in documents:
        # Build a crawl-path breadcrumb for provenance (with redacted URLs)
        redacted_path = [*map(_redact_url, doc.trace.path), _redact_url(doc.url)]
        provenance = " -> ".join(redacted_path)

        chunks = _chunk_text(doc.content, chunk_size)
        for i, chunk in enumerate(chunks):
            # Embed source metadata directly in the chunk content
            enriched = (
                f"Source: {_redact_url(doc.url)}\n"
                f"Crawl path: {provenance}\n"
                f"Depth: {doc.depth}\n\n"
                f"{chunk}"
            )

            filename = f"{doc.source_type.value}_{doc.content_hash}_chunk{i}.txt"
            buf = BytesIO(enriched.encode("utf-8"))
            buf.name = filename

            try:
                uploaded = client.files.create(file=buf, purpose="assistants")
                client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=uploaded.id,
                    chunking_strategy={
                        "type": "static",
                        "static": {"max_chunk_size_tokens": 512, "chunk_overlap_tokens": 64},
                    },
                )
                indexed += 1
            except Exception as exc:
                print(colored(f"  Failed to index chunk {i} of {doc.title}: {exc}", "red"))

    return indexed


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

# Default seed URLs for demonstration.
DEFAULT_SEEDS = [
    ("https://github.com/meta-llama/llama-stack", LinkType.GITHUB, "meta-llama/llama-stack"),
]


def _detect_seed_type(url: str) -> tuple[LinkType, str]:
    """Auto-detect the source type and identifier from a seed URL."""
    for link_type, pattern in LINK_PATTERNS.items():
        match = pattern.search(url)
        if match:
            identifier = _extract_identifier(link_type, match, url)
            return link_type, identifier
    return LinkType.WEB, url


def main(
    host: str,
    port: int,
    model_id: str | None = None,
    embedding_model_id: str | None = None,
    max_depth: int = 1,
    max_docs: int = 10,
    seed_urls: list[str] | None = None,
    question: str = "What is Llama Stack and what APIs does it provide?",
    system_prompt: str | None = None,
) -> None:
    """Run the multi-source RAG crawler demo.

    Args:
        host: Llama Stack server hostname.
        port: Llama Stack server port.
        model_id: Chat model to use (auto-detected if omitted).
        embedding_model_id: Embedding model (auto-detected if omitted).
        max_depth: How many link-hops to follow from seed documents.
        max_docs: Maximum total documents to crawl.
        seed_urls: Seed URLs to start crawling from. Supports GitHub, GitLab,
            Confluence, Jira, Google Docs, and web URLs. Source type is
            auto-detected. Defaults to the llama-stack GitHub repo.
        question: Question to answer using the crawled knowledge base.
        system_prompt: Custom system prompt for the RAG query. If omitted, a
            default prompt that instructs the model to cite sources and crawl
            paths is used.

    Environment variables for authenticated sources:
        GITHUB_TOKEN: GitHub personal access token (for private repos).
        GITLAB_TOKEN: GitLab personal access token.
        GITLAB_URL: GitLab base URL (auto-detected from seed URL if omitted).
        CONFLUENCE_URL: Confluence base URL (e.g. https://confluence.example.com).
        CONFLUENCE_TOKEN: Confluence personal access token.
        JIRA_URL: Jira base URL (e.g. https://jira.example.com).
        JIRA_TOKEN: Jira personal access token.
        GOOGLE_API_KEY: Google API key for Docs access.

    Examples:
        # GitHub (public, no auth needed)
        python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321

        # GitLab
        GITLAB_TOKEN=glpat-xxx python -m demos.03_rag.06_multi_source_rag_crawler \\
          localhost 8321 --seed_urls='["https://gitlab.com/myorg/myproject"]'

        # Confluence
        CONFLUENCE_URL=https://confluence.example.com CONFLUENCE_TOKEN=xxx \\
          python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321 \\
          --seed_urls='["https://confluence.example.com/wiki/spaces/TEAM/pages/12345"]'

        # Jira
        JIRA_URL=https://jira.example.com JIRA_TOKEN=xxx \\
          python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321 \\
          --seed_urls='["https://jira.example.com/browse/PROJ-100"]' \\
          --question="What is the status of PROJ-100?"

        # Google Docs
        GOOGLE_API_KEY=AIzaSy... python -m demos.03_rag.06_multi_source_rag_crawler \\
          localhost 8321 \\
          --seed_urls='["https://docs.google.com/document/d/1abc...xyz"]'

        # Mixed sources (crawl will follow links between them)
        python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321 \\
          --seed_urls='["https://github.com/org/repo","https://jira.example.com/browse/PROJ-1"]' \\
          --max_depth=2
    """
    _maybe_load_dotenv()

    # -- Connect to Llama Stack -------------------------------------------------
    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    resolved_model = model_id or os.environ.get("LLAMA_STACK_MODEL")
    if resolved_model is None:
        resolved_model = get_any_available_chat_model(client)
        if resolved_model is None:
            return
    else:
        if not check_model_is_available(client, resolved_model):
            return
        if not can_model_chat(client, resolved_model):
            print(colored(f"Model `{resolved_model}` does not support chat.", "red"))
            return

    embedding_model = embedding_model_id or get_any_available_embedding_model(client)
    if embedding_model is None:
        return

    embedding_dimension = get_embedding_dimension(client, embedding_model)
    if embedding_dimension is None:
        print(colored("Unable to determine embedding dimension.", "red"))
        return

    vector_providers = [p for p in client.providers.list() if p.api == "vector_io"]
    if not vector_providers:
        print(colored("No vector_io providers available.", "red"))
        return
    provider_id = vector_providers[0].provider_id

    # -- Phase 1: Crawl ---------------------------------------------------------
    print(colored("\n=== Phase 1: Crawling documents ===\n", "cyan"))
    print(f"  max_depth={max_depth}, max_docs={max_docs}")

    crawler = MultiSourceCrawler(max_depth=max_depth, max_docs=max_docs)

    # Register all available fetchers
    crawler.register_fetcher(LinkType.GITHUB, fetch_github)
    crawler.register_fetcher(LinkType.GITLAB, fetch_gitlab)
    crawler.register_fetcher(LinkType.CONFLUENCE, fetch_confluence)
    crawler.register_fetcher(LinkType.JIRA, fetch_jira)
    crawler.register_fetcher(LinkType.GOOGLE_DOCS, fetch_google_doc)
    crawler.register_fetcher(LinkType.WEB, fetch_web_page)

    # Show which sources have credentials configured
    available = ["github (public)"]
    if os.environ.get("GITHUB_TOKEN"):
        available[-1] = "github (authenticated)"
    if os.environ.get("GITLAB_TOKEN"):
        available.append("gitlab")
    if os.environ.get("CONFLUENCE_URL") and os.environ.get("CONFLUENCE_TOKEN"):
        available.append("confluence")
    if os.environ.get("JIRA_URL") and os.environ.get("JIRA_TOKEN"):
        available.append("jira")
    if os.environ.get("GOOGLE_API_KEY"):
        available.append("google_docs")
    available.append("web")
    print(f"  Available sources: {', '.join(available)}")

    # Add seeds with auto-detected source types
    if seed_urls:
        for url in seed_urls:
            link_type, identifier = _detect_seed_type(url)
            print(f"  Seed: {url} (detected as {link_type.value})")
            crawler.add_seed(url, link_type, identifier)
    else:
        for url, link_type, identifier in DEFAULT_SEEDS:
            print(f"  Seed: {url} ({link_type.value})")
            crawler.add_seed(url, link_type, identifier)

    print()
    documents = crawler.crawl()
    stats = crawler.stats

    print(colored(f"\n  Crawl complete: {stats.total_fetched} documents fetched", "green"))
    if stats.by_source:
        breakdown = ", ".join(f"{k}: {v}" for k, v in stats.by_source.items())
        print(f"  By source: {breakdown}")
    if stats.by_depth:
        breakdown = ", ".join(f"depth {k}: {v}" for k, v in sorted(stats.by_depth.items()))
        print(f"  By depth: {breakdown}")
    if stats.total_skipped:
        print(f"  Skipped (duplicates): {stats.total_skipped}")
    if stats.total_failed:
        print(f"  Failed: {stats.total_failed}")

    if not documents:
        print(colored("No documents were crawled. Check network access and seed URLs.", "red"))
        return

    # -- Phase 2: Index into vector store ----------------------------------------
    print(colored("\n=== Phase 2: Indexing into Llama Stack vector store ===\n", "cyan"))

    vector_store = None
    try:
        vector_store = client.vector_stores.create(
            name=f"crawler_demo_{uuid4()}",
            extra_body={
                "provider_id": provider_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
        )
        print(f"  Created vector store: {vector_store.id}")

        indexed = index_crawled_documents(client, vector_store.id, documents)
        print(colored(f"  Indexed {indexed} chunks from {len(documents)} documents", "green"))

        # -- Phase 3: Query with RAG ------------------------------------------------
        print(colored("\n=== Phase 3: Querying with RAG ===\n", "cyan"))
        print(f"  Question: {question}\n")

        default_instructions = (
            "Answer the question using the documents provided via file_search. "
            "Each document chunk includes a 'Source:' URL and a 'Crawl path:' showing "
            "how it was discovered (e.g. README -> linked doc -> linked issue). "
            "When referencing information, cite the source URL. "
            "If relevant, mention the crawl path to show how the information connects."
        )
        instructions = system_prompt or default_instructions
        print(f"  System prompt: {instructions[:100]}{'...' if len(instructions) > 100 else ''}\n")

        response = client.responses.create(
            model=resolved_model,
            instructions=instructions,
            input=[{"role": "user", "content": question}],
            tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
            tool_choice={"type": "file_search"},
            include=["file_search_call.results"],
            stream=False,
        )

        # Show which sources were retrieved (with scores and provenance)
        _print_retrieved_sources(response)

        print(colored("\n[response]", "green"), response.output_text or response.output)

    finally:
        if vector_store is not None:
            try:
                client.vector_stores.delete(vector_store_id=vector_store.id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete vector store: {e}", "yellow"))


if __name__ == "__main__":
    fire.Fire(main)
