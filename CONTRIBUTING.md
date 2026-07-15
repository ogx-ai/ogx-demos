# Contributing to ogx-demos

We welcome demo contributions from OGX feature engineers. When a new feature lands, the feature owner is encouraged to add a corresponding demo to this repository.

For demo structure, naming conventions, and documentation standards, see [DEMOS_STRUCTURE.md](DEMOS_STRUCTURE.md).

## Quick Start: Adding a Demo

1. **Fork and clone** the repository, then create a branch:

   ```bash
   git checkout -b <your-username>/<short-description> upstream/main
   ```

2. **Pick the right phase directory** under `demos/`. See [DEMOS_STRUCTURE.md](DEMOS_STRUCTURE.md) for phase descriptions. If your demo doesn't fit an existing phase, open an issue to discuss adding a new one.

3. **Copy the template**:

   ```bash
   cp demos/_template.py demos/<phase>/NN_your_demo.py
   ```

   Use a numbered prefix (`NN_`) that places your demo in the correct learning order within the phase. Check existing files in the directory for the next available number.

4. **Implement your demo** following the template. Key patterns:
   - Docstring with `Demo:`, `Description:`, and `Learning Objectives:` fields.
   - `main(host, port, ...)` function with `fire.Fire(main)`.
   - Use `resolve_model(client, model_id)` from `demos.shared.utils` for model selection.
   - Use `from demos.shared.utils import ...` for shared utilities (not `sys.path` manipulation).
   - Clean up any resources (vector stores, files) in a `finally` block.

5. **Declare runtime dependencies** with `# demo-requires:` comment tags at the top of the file, before the docstring:

   ```python
   # demo-requires: TAVILY_SEARCH_API_KEY
   # demo-requires: vision_model
   """
   Demo: My Demo
   ...
   """
   ```

   - `UPPER_CASE` values are treated as environment variable names. The test harness (`tests/run_demos.py`) checks whether these are set before running the demo.
   - `lower_case` values are capability tags (e.g., `vision_model`, `mcp_server`).
   - Demos without any `# demo-requires:` tags are assumed to have no special dependencies.
   - If your demo requires a new environment variable that isn't in `.env.example`, add it there (commented out, with a description). Users can then uncomment and fill in their own keys in their local `.env` to enable the demo in the test harness.

6. **Update the phase README** (`demos/<phase>/README.md`) with an entry for your demo, following the format of existing entries.

7. **Run checks locally**:

   ```bash
   # Lint and validate
   pre-commit run --all-files

   # Run your demo against a local server
   uv run python -m demos.<phase>.<demo_name> localhost 8321

   # Run the test harness (optional, requires a running OGX server)
   uv run python tests/run_demos.py localhost 8321 --demo demos/<phase>/<demo_name>.py
   ```

8. **Open a pull request** against `main`.

## Checklist

Before opening your PR, verify:

- [ ] Demo has the standard docstring (`Demo:` / `Description:` / `Learning Objectives:`)
- [ ] Uses `fire.Fire(main)` with `main(host: str, port: int, ...)` signature
- [ ] Uses package imports (`from demos.shared.utils import ...`)
- [ ] Uses `resolve_model(client, model_id)` for model selection (not `os.getenv("OGX_MODEL")`)
- [ ] Runtime dependencies declared with `# demo-requires:` tags
- [ ] Resources are cleaned up in `finally` blocks
- [ ] Phase `README.md` updated with new demo entry
- [ ] `pre-commit run --all-files` passes

## Shared Utilities

Common helper functions live in `demos/shared/utils.py`. Use them instead of duplicating logic across demos (e.g., `resolve_model`, `resolve_embedding_model`, `get_embedding_dimension`).

**When to add to shared utilities**: if 3 or more demos would benefit from the same helper function, add it to `demos/shared/utils.py`. Otherwise, keep the logic local to your demo.

## Adding a New Phase

Adding a new top-level phase directory (e.g., `demos/08_new_topic/`) is rare. If you believe one is needed:

1. Create the directory with an `__init__.py` and a `README.md` following the format of existing phase READMEs.
2. Update the root `README.md` and `DEMOS_STRUCTURE.md` to include the new phase.
3. Explain the scope and placement in the PR description.

## Running Demos

All demos are run as Python modules:

```bash
uv run python -m demos.<phase>.<demo_name> <host> <port> [options]
```

For example:

```bash
uv run python -m demos.01_foundations.01_client_setup localhost 8321
```

## Questions?

Open an issue in the [repository](https://github.com/ogx-ai/ogx-demos/issues).
