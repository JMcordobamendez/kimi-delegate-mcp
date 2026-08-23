"""Tests for the guards that stand between a delegated model and the filesystem.

These run offline: `_call_kimi` is the only network seam and every test that
needs a reply injects one. That is the same rule the server asks delegated code
to follow — side effects behind an injectable parameter — applied to itself.

    python -m pytest tests/ -q
"""
import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import server  # noqa: E402


@pytest.fixture
def base(tmp_path):
    """A project root to confine writes to."""
    (tmp_path / "ok").mkdir()
    return tmp_path


@pytest.fixture
def reply(monkeypatch):
    """Inject a canned Kimi reply, bypassing the network entirely."""
    def use(text, finish="stop", footer="_footer_"):
        monkeypatch.setattr(server, "_call_kimi", lambda sp, p: (text, footer, finish))
    return use


# --------------------------------------------------------------------------
# Path confinement. The model chooses these strings, so they are hostile input.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rel", [
    "file.py",
    "ok/file.py",
    "a/b/c/deep.py",          # missing parents are created
    "src/sub/../other.py",    # '..' that stays inside is fine
    "\x1b[1;34mapp.py\x1b[0m",  # ANSI stripped, then checked
    "`quoted.py`",
])
def test_paths_inside_base_are_accepted(base, rel):
    assert server._resolve_write_path(base, rel).is_relative_to(base)


@pytest.mark.parametrize("rel,reason", [
    ("../escape.py",       "outside"),
    ("../../etc/passwd",   "outside"),
    ("/etc/passwd",        "absolute"),
    ("~/x.py",             "absolute"),
    (".env",               "sensitive"),
    ("a/credentials.json", "sensitive"),
    ("id_rsa",             "sensitive"),
    ("deploy.pem",         "sensitive"),
    ("",                   "empty"),
    ("\x1b[0m",            "empty"),   # cleans down to nothing
    ("   ",                "empty"),
])
def test_paths_outside_or_sensitive_are_refused(base, rel, reason):
    with pytest.raises(ValueError) as e:
        server._resolve_write_path(base, rel)
    assert reason in str(e.value).lower()


def test_symlink_escaping_base_is_refused(base):
    """Documented limitation, asserted so it cannot regress silently."""
    outside = base.parent / "outside"
    outside.mkdir(exist_ok=True)
    (base / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        server._resolve_write_path(base, "link/x.py")


# --------------------------------------------------------------------------
# The workflow guard: commits, pushes and publishing belong to the human.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "pytest -q", "python -m pytest", "ls -la", "cat README.md",
    "git status", "git diff", "git log --oneline",   # read-only git stays usable
])
def test_harmless_commands_are_allowed(cmd):
    assert not server._FORBIDDEN_CMD_RE.search(cmd)


@pytest.mark.parametrize("cmd", [
    "git commit -m x", "git push origin main", "git tag v1",
    "gh pr create", "gh release upload",
    "npm publish", "pnpm publish", "twine upload dist/*", "cargo publish",
    "git reset --hard", "git checkout .", "git clean -fd", "git stash",
    "ls && git push",                 # chained
    "echo hi; git commit -m x",       # chained with ;
    "GIT_DIR=. git commit -m x",      # prefixed with an env var
])
def test_publishing_and_destructive_commands_are_refused(cmd):
    assert server._FORBIDDEN_CMD_RE.search(cmd)


# --------------------------------------------------------------------------
# Session ids come back as an argument, so they are untrusted.
# --------------------------------------------------------------------------

def test_wellformed_session_id_maps_into_the_session_dir():
    p = server._session_path("a" * 32)
    assert p.parent == server.SESSION_DIR


@pytest.mark.parametrize("sid", [
    "../../etc/passwd", "../x", "A" * 32, "abc", "", "a" * 31, "a" * 33, "a/b",
])
def test_malformed_session_id_is_refused(sid):
    with pytest.raises(ValueError):
        server._session_path(sid)


# --------------------------------------------------------------------------
# Output clipping must keep the verdict, which runners put last.
# --------------------------------------------------------------------------

def test_clip_keeps_the_tail_for_shell_output():
    text = "\n".join(f"line {i}" for i in range(5000)) + "\nFAILED test_x"
    out = server._clip(text, keep_tail=True)
    assert "FAILED test_x" in out
    assert out.startswith("line 0")


def test_clip_leaves_short_text_untouched():
    assert server._clip("short") == "short"


# --------------------------------------------------------------------------
# A written .py file that does not even parse must not report as success.
# --------------------------------------------------------------------------

def test_syntax_error_is_reported_for_python():
    msg = server._syntax_error(pathlib.Path("a.py"), "def f():\n    return 1\n}\n")
    assert "does not parse" in msg


@pytest.mark.parametrize("name,text", [
    ("a.py", "def f():\n    return 1\n"),
    ("a.py", ""),
    ("a.txt", "def f(:"),      # not Python: not our business
    ("a.md", "}}}}"),
])
def test_no_syntax_complaint_when_not_applicable(name, text):
    assert server._syntax_error(pathlib.Path(name), text) == ""


# --------------------------------------------------------------------------
# Prompt layout. Caching is by prefix, so the varying part must come last.
# --------------------------------------------------------------------------

def test_task_comes_last_so_the_prefix_stays_stable():
    prompt = server._build_prompt("THE_TASK", [], "THE_CONTEXT")
    assert prompt.index("THE_CONTEXT") < prompt.index("THE_TASK")
    assert prompt.rstrip().endswith("THE_TASK")


def test_context_files_are_sorted_so_ordering_cannot_break_the_prefix(base, monkeypatch):
    for n in ("b.py", "a.py", "c.py"):
        (base / n).write_text(f"# {n}\n")
    monkeypatch.chdir(base)
    prompt = server._build_prompt("t", ["c.py", "a.py", "b.py"], "")
    assert prompt.index("a.py") < prompt.index("b.py") < prompt.index("c.py")


# --------------------------------------------------------------------------
# delegate_and_apply, end to end with an injected reply.
# --------------------------------------------------------------------------

def test_files_are_written_and_summarised(base, reply):
    reply("FILE: ok/x.py\n```python\nx = 1\n```\n")
    out = server.delegate_and_apply(task="t", base_dir=str(base))
    assert (base / "ok" / "x.py").read_text() == "x = 1\n"
    assert "ok/x.py" in out


def test_truncated_reply_writes_nothing(base, reply):
    reply("FILE: a.py\n```python\nx = 1\n```\n", finish="length")
    out = server.delegate_and_apply(task="t", base_dir=str(base))
    assert "ABORTED" in out
    assert not (base / "a.py").exists(), "a truncated reply must not reach disk"


def test_unparseable_reply_writes_nothing_and_returns_the_text(base, reply):
    reply("I could not do that, sorry.")
    out = server.delegate_and_apply(task="t", base_dir=str(base))
    assert "No 'FILE:' block" in out
    assert "I could not do that" in out
    assert list(base.glob("*.py")) == []


def test_escaping_path_is_refused_while_valid_siblings_are_written(base, reply):
    reply(
        "FILE: good.py\n```python\ng = 1\n```\n\n"
        "FILE: ../evil.py\n```python\ne = 1\n```\n"
    )
    out = server.delegate_and_apply(task="t", base_dir=str(base))
    assert (base / "good.py").exists()
    assert not (base.parent / "evil.py").exists()
    assert "Refused" in out


def test_broken_python_is_written_but_flagged(base, reply):
    reply("FILE: bad.py\n```python\ndef f():\n    return 1\n}\n```\n")
    out = server.delegate_and_apply(task="t", base_dir=str(base))
    assert (base / "bad.py").exists(), "still written; git is the undo path"
    assert "does not parse" in out


def test_nonexistent_base_dir_is_an_error(reply):
    reply("FILE: a.py\n```python\nx = 1\n```\n")
    out = server.delegate_and_apply(task="t", base_dir="/nope/does/not/exist")
    assert "ERROR" in out


# --------------------------------------------------------------------------
# The agentic loop refuses to start on a dirty tree, since git is the undo.
# --------------------------------------------------------------------------

def test_agentic_refuses_a_non_git_directory(base):
    out = server.delegate_agentic(task="t", base_dir=str(base), max_turns=1)
    assert "ABORTED" in out


# --------------------------------------------------------------------------
# The prompts sent to the model are English-only; a stray translation would
# change what the model receives without anyone noticing.
# --------------------------------------------------------------------------

# Spanish-specific characters. Typographic marks like the em dash are fine —
# the point is to catch a prompt drifting back to another language, not to ban
# punctuation.
_NON_ENGLISH = set("ñáéíóúüÑÁÉÍÓÚÜ¿¡")


@pytest.mark.parametrize("prompt_name", [
    "TESTABILITY_RULE", "SYSTEM_PROMPT", "SYSTEM_PROMPT_APPLY", "SYSTEM_PROMPT_AGENTIC",
])
def test_prompts_stay_in_english(prompt_name):
    """These are sent to the model, so a stray translation changes its input."""
    found = _NON_ENGLISH & set(getattr(server, prompt_name))
    assert not found, f"{prompt_name} contains {sorted(found)}"


def test_tool_descriptions_stay_in_english():
    for tool in server.AGENTIC_TOOLS:
        desc = tool["function"]["description"]
        found = _NON_ENGLISH & set(desc)
        assert not found, f"{tool['function']['name']} contains {sorted(found)}"


def test_server_module_has_no_syntax_errors():
    src = pathlib.Path(server.__file__).read_text()
    ast.parse(src)
