import re
from dataclasses import dataclass
from pathlib import Path

from repopilot.config import config
from repopilot.observability.logger import get_logger
from tree_sitter_languages import get_parser

logger = get_logger(__name__)


# Maps file extension → tree-sitter language name.
# tree-sitter-languages bundles all these grammars, no extra installs needed.
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
}

# Text/config files have no meaningful AST — we chunk them by line count instead.
TEXT_EXTENSIONS = {".md", ".txt", ".yaml", ".yml", ".json", ".toml"}
ALL_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys()) | TEXT_EXTENSIONS

# Sliding window settings for text files and AST fallback.
# CHUNK_OVERLAP ensures context isn't lost at chunk boundaries.
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

# Tree-sitter node type names that correspond to a named, indexable block.
# These are consistent across languages — tree-sitter uses the same names where possible.
BLOCK_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "arrow_function",
    "class_definition",
    "class_declaration",
    "method_declaration",
    "constructor_declaration",
    "interface_declaration",
    "function_item",  # Rust uses this instead of function_definition
    "func_declaration",  # Go
}

# Bumped whenever chunking changes. Indexers store it with every chunk and
# re-chunk files indexed by an older version, so an existing index upgrades itself.
#   2: module-level code (constants, config tables, header comments) is indexed too
CHUNKER_VERSION = 2

_MODULE_CHUNK_LINES = 60
_IMPORT = re.compile(r"^\s*(import |from \S+ import |#include |using |package |require\()")
_ASSIGNED = re.compile(r"^(?:export\s+)?(?:const\s+|let\s+|var\s+)?([A-Za-z_]\w*)\s*(?::[^=\n]+)?=")


@dataclass
class ParsedChunk:
    name: str
    type: str  # "function" | "class" | "module" | "block"
    content: str
    source: str  # absolute path to the file
    start_line: int
    end_line: int


def parse_file(filepath: str) -> list[ParsedChunk]:
    """Entry point — routes to AST parsing or sliding window based on file type."""
    ext = Path(filepath).suffix.lower()

    source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    if not source.strip():
        # e.g. an empty __init__.py — nothing to index, and not an error
        logger.debug(f"Skipping empty file: {filepath}")
        return []

    if ext in TEXT_EXTENSIONS:
        return _sliding_window(source.splitlines(), filepath)

    language_name = EXTENSION_TO_LANGUAGE.get(ext)
    if not language_name:
        raise ValueError(f"Unsupported file type: {ext}")

    return _parse_with_treesitter(source, filepath, language_name)


def _parse_with_treesitter(source: str, filepath: str, language_name: str) -> list[ParsedChunk]:
    """Parse source with the appropriate tree-sitter grammar and extract named blocks."""
    logger.info(f"Parsing {language_name} file: {filepath}")
    parser = get_parser(language_name)
    # tree-sitter positions are BYTE offsets, so slice the encoded bytes, never the str —
    # otherwise any multi-byte character (—, ✓, é…) shifts every chunk after it.
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    lines = source.splitlines()

    chunks = []
    _walk(tree.root_node, source_bytes, filepath, chunks, depth=0)

    # If the AST yielded nothing (e.g. a file with only imports), fall back to line chunks
    if not chunks:
        logger.warning(f"No AST blocks found in {filepath}, falling back to sliding window")
        return _sliding_window(lines, filepath)

    chunks += _module_chunks(lines, chunks, filepath)
    chunks.sort(key=lambda c: c.start_line)
    logger.info(f"Parsed {len(chunks)} chunks from {filepath}")
    return chunks


def _module_chunks(lines: list[str], blocks: list[ParsedChunk], filepath: str) -> list[ParsedChunk]:
    """
    Chunks for the code between functions and classes: constants, config tables,
    header comments. Without them a question about e.g. which commands are always
    refused could never retrieve _ALWAYS_DENIED, and the agent repeated the same
    search up to 12 times in /eval. Runs of import lines, decorators and blank
    lines alone are skipped.
    """
    covered = set()
    for block in blocks:
        covered.update(range(block.start_line, block.end_line + 1))

    segments, current = [], []
    for number, text in enumerate(lines, 1):
        if number in covered:
            if current:
                segments.append(current)
            current = []
        else:
            current.append((number, text))
    if current:
        segments.append(current)

    chunks = []
    for segment in segments:
        for i in range(0, len(segment), _MODULE_CHUNK_LINES):
            part = segment[i : i + _MODULE_CHUNK_LINES]
            while part and not part[0][1].strip():
                part = part[1:]
            while part and not part[-1][1].strip():
                part = part[:-1]
            meaningful = [t for _, t in part if t.strip() and not _IMPORT.match(t) and not t.lstrip().startswith("@")]
            code = [t for t in meaningful if not t.lstrip().startswith(("#", "//", "*", "/*"))]
            prose = [t for t in meaningful if t not in code and re.search(r"[A-Za-z]{3}", t)]
            if not code and len(prose) < 3:
                continue  # nothing but a section divider like "# --- Files ---"
            names = [m.group(1) for t in meaningful if (m := _ASSIGNED.match(t))]
            chunks.append(
                ParsedChunk(
                    name=", ".join(dict.fromkeys(names[:3])) or "module code",
                    type="module",
                    content="\n".join(t for _, t in part),
                    source=filepath,
                    start_line=part[0][0],
                    end_line=part[-1][0],
                )
            )
    return chunks


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _walk(node, source: bytes, filepath: str, chunks: list, depth: int):
    """
    Recursively walk the AST. When a named block node is found, record it and stop
    descending — this keeps chunks at the top level and avoids duplicating nested functions.
    """
    if node.type in BLOCK_NODE_TYPES:
        name = _extract_name(node, source)
        content = _node_text(node, source)
        chunk_type = "class" if "class" in node.type else "function"
        chunks.append(
            ParsedChunk(
                name=name,
                type=chunk_type,
                content=content,
                source=filepath,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )
        logger.debug(f"  Found {chunk_type} '{name}' (lines {node.start_point[0] + 1}-{node.end_point[0] + 1})")
        return  # stop here — don't index nested functions/classes as separate chunks

    for child in node.children:
        _walk(child, source, filepath, chunks, depth + 1)


def _extract_name(node, source: bytes) -> str:
    """Find the identifier child of a block node and return its text as the chunk name."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return _node_text(child, source)
    return node.type  # fallback to node type if no name found


def _sliding_window(lines: list[str], filepath: str) -> list[ParsedChunk]:
    """
    Split lines into overlapping fixed-size chunks.
    Used for text files and as a fallback when AST parsing finds nothing.
    """
    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i, start in enumerate(range(0, len(lines), step)):
        end = min(start + CHUNK_SIZE, len(lines))
        text = "\n".join(lines[start:end]).strip()
        if text:
            chunks.append(
                ParsedChunk(
                    name=f"chunk_{i}",
                    type="block",
                    content=text,
                    source=filepath,
                    start_line=start + 1,
                    end_line=end,
                )
            )
        if end == len(lines):
            break

    logger.info(f"Parsed {len(chunks)} chunks from {filepath}")
    return chunks


# Directories that never contain the user's own source code.
# .repopilot / .chromadb hold this tool's own data and must not be indexed.
SKIP_DIRS = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".git",
    "node_modules",
    "dist",
    "build",
    ".repopilot",
    ".chromadb",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".idea",
    ".vscode",
}

# Generated lock files — huge, noisy, and never useful as search results.
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "composer.lock", "Cargo.lock"}

# Files bigger than this are almost always generated or data, not code.
MAX_FILE_BYTES = 500_000


def _is_eval_questions_file(path: Path) -> bool:
    questions_file = config.get("eval", {}).get("questions_file", "data/evals/questions.yaml")
    # leading "/" so a relative path ("data/evals/questions.yaml" itself) matches too
    normalised = "/" + str(path).replace("\\", "/").lower()
    return normalised.endswith("/" + questions_file.replace("\\", "/").lower().removeprefix("./"))


def is_indexable(path: str | Path, root: str | Path | None = None, skip_dirs: set[str] | None = None) -> bool:
    """
    True if path is a source/text file worth indexing (shared by the full scan and the watcher).
    Skip-dir matching only looks at the part of the path below root, so a repo that
    itself lives under e.g. ~/build/ is still indexed.
    """
    path = Path(path)
    if path.suffix.lower() not in ALL_EXTENSIONS:
        return False
    if path.name in SKIP_FILES or path.name.startswith(".env"):
        return False
    if _is_eval_questions_file(path):
        # The eval set contains expected answers — indexing it would let /ask
        # "find" them during /eval and inflate the scores.
        return False
    parts = path.parts
    if root is not None:
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            pass
    if any(part in (skip_dirs or SKIP_DIRS) or part.endswith(".egg-info") for part in parts):
        return False
    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        # Deleted/moved files can't be stat'ed — let the caller decide (e.g. watcher deletes)
        return True


def get_source_files(repo_path: str, skip_dirs: list[str] = None) -> list[str]:
    """Recursively find all indexable source files under repo_path, skipping build/env dirs."""
    skip = set(skip_dirs) if skip_dirs else SKIP_DIRS
    files = [
        str(path)
        for path in Path(repo_path).rglob("*")
        if path.is_file() and is_indexable(path, root=repo_path, skip_dirs=skip)
    ]
    logger.info(f"Found {len(files)} source files in {repo_path}")
    return files
