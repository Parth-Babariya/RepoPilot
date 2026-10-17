---
name: debug_error
description: Find the root cause of an error message, exception or traceback
when_to_use: the user shares an error, exception, traceback or crash, or asks why X fails
---
Find the root cause in the code, not just the line that raised.

1. Read the error. Note the exception type and message. In a traceback, start from the
   innermost frame that is in this project (skip library frames).
2. Call find_symbol on that function and read the failing line. Work out which value
   was wrong or missing.
3. Trace that value back to where it comes from: call find_callers and follow the
   arguments up through the frames; use search_codebase to find where it is created,
   configured or loaded.
4. If it may be environmental (a missing setting, file or key, or a version change), look
   at the config files, and at recent changes with run_command `git log -n 5 -- <file>`.

Answer format:
- **Root cause** — one sentence.
- **Evidence** — the chain of locations (file:line) from the cause to the error.
- **Fix** — the specific change to make.
- **Check** — a test or command that confirms the fix.

If you can't confirm the cause, give the two most likely explanations and what would
tell them apart. Don't present a guess as the answer.
