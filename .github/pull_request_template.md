## What

<!-- Which demo(s) are you adding or modifying? -->

## Why

<!-- What OGX feature does this demo showcase? Link to the feature PR/issue if applicable. -->

## Phase

<!-- Which phase directory? e.g., 03_rag -->

## Checklist

- [ ] Demo follows the template structure (docstring, `fire.Fire(main)`, `main(host, port, ...)`)
- [ ] Uses `resolve_model(client, model_id)` for model selection
- [ ] Runtime dependencies declared with `# demo-requires:` tags
- [ ] Phase README updated with new demo entry
- [ ] Demo runs successfully against a local OGX server
- [ ] Resources cleaned up in `finally` blocks
- [ ] `pre-commit run --all-files` passes
