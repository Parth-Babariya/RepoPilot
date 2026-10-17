---
name: review_code
description: Review a file or function for bugs, risks and unclear code
when_to_use: requests to review code, find bugs in X, check whether X is correct, or what could go wrong with X
---
Review the code the user points to, using its real callers to judge what inputs it gets.

1. Read the target with find_symbol (or search_codebase if they name a file or feature).
2. Call find_callers to see how it is used — the inputs it really receives decide which
   edge cases matter.
3. Check, in this order:
   - **Correctness** — empty / None / zero / very large inputs, off-by-one, wrong conditions,
     state left inconsistent after an error.
   - **Error handling** — exceptions swallowed or too broad, resources not closed, retries
     that never stop.
   - **Security** — unvalidated input reaching SQL, shell commands or file paths; secrets in code or logs.
   - **Performance** — repeated work inside loops, queries in loops, unbounded growth.
   - **Clarity** — misleading names, duplicated logic, comments that contradict the code.

Answer format: findings ordered by severity (High / Medium / Low). For each give the
location (file:line), the problem, a concrete scenario that triggers it, and the fix in
words or a short snippet. RepoPilot does not edit code, so describe fixes — don't claim to
apply them. If you find nothing significant, say so; never invent issues to fill the list.
