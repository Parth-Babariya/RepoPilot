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
