from repopilot.context.indexers.code_parser import parse_file


def test_multibyte_characters_do_not_shift_chunks(tmp_path):
    # "—" and "✓" are 3 bytes each in UTF-8; tree-sitter reports byte offsets,
    # so slicing the str instead of the bytes used to garble names and content.
    source = (
        '"""Module docstring — with an em dash ✓"""\n'
        "\n"
        "def get_or_create_index():\n"
        "    return 1\n"
        "\n"
        "class Store:\n"
        "    def fail_task(self):\n"
        "        pass\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")

    chunks = {c.name: c for c in parse_file(str(path))}
    assert set(chunks) == {"get_or_create_index", "Store"}
    assert chunks["get_or_create_index"].content == "def get_or_create_index():\n    return 1"
    assert chunks["Store"].content.startswith("class Store:")
    assert chunks["Store"].start_line == 6


def test_empty_files_are_skipped_quietly(tmp_path):
    for name in ("__init__.py", "notes.md"):
        path = tmp_path / name
        path.write_text("  \n", encoding="utf-8")
        assert parse_file(str(path)) == []


def test_eval_answers_and_build_metadata_are_not_indexed(tmp_path, monkeypatch):
    from repopilot.context.indexers.code_parser import get_source_files

    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "questions.yaml").write_text("- question: q\n")
    (tmp_path / "pkg.egg-info").mkdir()
    (tmp_path / "pkg.egg-info" / "requires.txt").write_text("x\n")
    (tmp_path / "app.py").write_text("x = 1\n")

    monkeypatch.chdir(tmp_path)
    for root in (".", str(tmp_path)):          # relative and absolute roots behave the same
        files = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in get_source_files(root)}
        assert files == {"app.py"}
