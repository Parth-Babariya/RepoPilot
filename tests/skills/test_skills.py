from repopilot.skills.registry import SkillRegistry
from repopilot.skills.skill_tools import BUILTIN_SKILLS_DIR


def write_skill(root, name, description, body="Do the thing."):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")


def test_builtin_skills_load_and_are_well_formed():
    registry = SkillRegistry([BUILTIN_SKILLS_DIR])
    registry.load()

    assert set(registry.skill_names) == {"trace_flow", "review_code", "debug_error"}
    prompt = registry.build_skills_prompt()
    for name in registry.skill_names:
        assert f"- {name}:" in prompt  # listed with its description
        assert "when_to_use" in prompt
        assert len(registry.load_skill(name)) > 200  # has real instructions


def test_repo_skills_add_to_and_override_builtins(tmp_path):
    write_skill(tmp_path, "release_notes", "Draft release notes from recent commits")
    write_skill(tmp_path, "review_code", "Our team's review checklist", body="Team checklist.")

    registry = SkillRegistry([BUILTIN_SKILLS_DIR, tmp_path / "missing", tmp_path])
    registry.load()

    assert {"trace_flow", "debug_error", "release_notes", "review_code"} == set(registry.skill_names)
    assert registry.load_skill("review_code") == "Team checklist."  # repo's version wins


def test_no_skills_means_empty_prompt(tmp_path):
    registry = SkillRegistry([tmp_path / "nothing_here"])
    registry.load()
    assert registry.skill_names == [] and registry.build_skills_prompt() == ""
