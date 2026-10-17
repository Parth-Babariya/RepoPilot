---
name: trace_flow
description: Explain how a feature or request flows through the code, step by step
when_to_use: questions like how does X work end to end, walk me through X, what happens when X runs
---
Trace the flow through the real code — never describe steps you did not find.

1. Find the entry point. If the user names a function or class, call find_symbol on it.
   Otherwise call search_codebase with the feature name to find where it starts
   (a command handler, route, CLI entry or public function).
2. Follow the calls. Call find_callees on the entry function. For each callee that belongs
   to this project (not a library), call find_symbol to read it and continue from there.
   Stop after about four levels, or when you reach storage, I/O or an external API.
3. Note the branches: error handling, retries, early returns and anything that is skipped
   under some condition.

Answer format:
- **Flow** — numbered steps, each as `function` (file:line) — what it does in one line.
- **State it touches** — databases, files, caches or globals read or written.
- **Failure paths** — what happens when a step fails.

Cite a location for every step. If part of the flow is unclear (dynamic dispatch,
configuration-driven choice), say so instead of guessing.
