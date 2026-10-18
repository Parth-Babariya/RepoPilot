"""
The layer rule from the README, enforced: a module may import from its own
layer or any layer below it, never from a layer above.

    main → workflows / evaluation → agents → tools · retrieval · memory · skills · protocols
         → guardrails · services · observability · config
"""

import re
from pathlib import Path

LAYER = {
    "main": 0,
    "workflows": 1,
    "evaluation": 1,
    "agents": 2,
    "tools": 3,
    "retrieval": 3,
    "memory": 3,
    "skills": 3,
    "protocols": 3,
    "guardrails": 4,
    "services": 4,
    "observability": 4,
    "config": 4,
}
SRC = Path(__file__).resolve().parents[1] / "src"
IMPORT = re.compile(r"^\s*(?:from|import)\s+repopilot\.(\w+)", re.MULTILINE)


def test_every_src_folder_has_a_layer():
    folders = {p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("__")}
    assert folders <= set(LAYER), f"add new folders to LAYER: {folders - set(LAYER)}"


def test_no_module_imports_from_a_higher_layer():
    upward = []
    for path in SRC.rglob("*.py"):
        own = path.relative_to(SRC).parts[0].removesuffix(".py")
        if own not in LAYER:
            continue
        for target in IMPORT.findall(path.read_text(encoding="utf-8")):
            if target in LAYER and LAYER[target] < LAYER[own]:
                upward.append(f"{path.relative_to(SRC)} imports repopilot.{target}")
    assert not upward, upward
