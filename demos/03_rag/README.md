# RAG (Retrieval-Augmented Generation)

## Overview
This folder teaches Retrieval-Augmented Generation (RAG) techniques using Llama Stack's vector stores and search capabilities. These examples show how to ground model responses in retrieved documents, combine multiple data sources, and optimize retrieval strategies.

## Learning Objectives
- Build basic RAG systems with file_search tool
- Combine multiple vector stores for multi-source retrieval
- Filter search results using metadata attributes
- Optimize chunking strategies for different use cases
- Implement hybrid search combining local and web sources
- Crawl and index documents across platforms with link following

## Demos

### Demo 1: Simple RAG
**File**: `01_simple_rag.py`

**Concepts**: RAG workflow (embed, store, retrieve, generate), file_search tool, vector store creation, document upload, context retrieval

**Description**: Demonstrates the basic RAG workflow by creating a vector store, uploading a document, and using the file_search tool to retrieve relevant context for answering questions.

**Run**:
```bash
python -m demos.03_rag.01_simple_rag localhost 8321
```

### Demo 2: Multi-Source RAG
**File**: `02_multi_source_rag.py`

**Concepts**: Multiple vector stores, cross-source retrieval, diverse knowledge bases, consolidated search

**Description**: Shows how to perform RAG across two separate vector stores using file_search, enabling retrieval from multiple knowledge sources simultaneously.

**Run**:
```bash
python -m demos.03_rag.02_multi_source_rag localhost 8321
```

### Demo 3: RAG with Metadata
**File**: `03_rag_with_metadata.py`

**Concepts**: Metadata filtering, document attributes, filtered search, metadata-based retrieval, source filtering

**Description**: Demonstrates how to filter file_search results by metadata attributes, allowing for more targeted retrieval based on document properties like source, date, or category.

**Run**:
```bash
python -m demos.03_rag.03_rag_with_metadata localhost 8321 --source doc_a
```

### Demo 4: Chunking Strategies
**File**: `04_chunking_strategies.py`

**Concepts**: Chunking strategies, chunk size optimization, chunk overlap, static chunking, retrieval quality optimization

**Description**: Compares different chunk sizes for the same document to demonstrate how chunking strategies affect retrieval quality and response accuracy.

**Run**:
```bash
python -m demos.03_rag.04_chunking_strategies localhost 8321
```

### Demo 5: Hybrid Search
**File**: `05_hybrid_search.py`

**Concepts**: Hybrid search, file_search + web_search combination, local and external knowledge, multi-source synthesis, context aggregation

**Description**: Combines file_search (for local vector store retrieval) with web_search (for real-time web data) in a single workflow, synthesizing information from both local documents and the web.

**Run**:
```bash
python -m demos.03_rag.05_hybrid_search localhost 8321
```

### Demo 6: Multi-Source RAG Crawler
**File**: `06_multi_source_rag_crawler.py`

**Concepts**: Cross-source crawling, link following, breadth-first discovery, crawl provenance, content deduplication, multi-platform RAG

**Description**: Builds a document crawler that starts from seed URLs and follows links across platforms (GitHub, GitLab, Confluence, Jira, Google Docs, web pages). Discovered documents are indexed into a Llama Stack vector store with provenance metadata, enabling RAG queries across an automatically-discovered knowledge graph. Source type is auto-detected from seed URLs. Fetchers for authenticated sources (GitLab, Confluence, Jira, Google Docs) activate automatically when credentials are set via environment variables.

**Supported Sources**:
| Source | Auth Env Vars | Seed URL Example |
|--------|--------------|------------------|
| GitHub | `GITHUB_TOKEN` (optional, for private repos) | `https://github.com/org/repo` |
| GitLab | `GITLAB_TOKEN` (required) | `https://gitlab.com/org/project` |
| Confluence | `CONFLUENCE_URL` + `CONFLUENCE_TOKEN` | `https://confluence.example.com/wiki/spaces/TEAM/pages/12345` |
| Jira | `JIRA_URL` + `JIRA_TOKEN` | `https://jira.example.com/browse/PROJ-123` |
| Google Docs | `GOOGLE_API_KEY` | `https://docs.google.com/document/d/1abc...xyz` |
| Web pages | *(none)* | `https://example.com/docs/guide.md` |

**Run**:
```bash
# GitHub (public, no auth needed)
python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321

# Customize crawl depth and document limits
python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321 --max_depth=2 --max_docs=20

# GitLab
GITLAB_TOKEN=glpat-xxx python -m demos.03_rag.06_multi_source_rag_crawler \
  localhost 8321 --seed_urls='["https://gitlab.com/myorg/myproject"]'

# Confluence
CONFLUENCE_URL=https://confluence.example.com CONFLUENCE_TOKEN=xxx \
  python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321 \
  --seed_urls='["https://confluence.example.com/wiki/spaces/TEAM/pages/12345"]'

# Jira
JIRA_URL=https://jira.example.com JIRA_TOKEN=xxx \
  python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321 \
  --seed_urls='["https://jira.example.com/browse/PROJ-100"]' \
  --question="What is the status of PROJ-100?"

# Google Docs
GOOGLE_API_KEY=AIzaSy... python -m demos.03_rag.06_multi_source_rag_crawler \
  localhost 8321 --seed_urls='["https://docs.google.com/document/d/1abc...xyz"]'

# Mixed sources -- crawler follows links between them
python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321 \
  --seed_urls='["https://github.com/org/repo","https://jira.example.com/browse/PROJ-1"]' \
  --max_depth=2

# Custom system prompt for the RAG query
python -m demos.03_rag.06_multi_source_rag_crawler localhost 8321 \
  --system_prompt="Answer concisely in bullet points. Always cite source URLs."
```
