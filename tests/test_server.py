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
# Running out of turns. The loop used to stop dead and throw the context away:
# the tree was left half-edited and the retry re-paid for the exploration, which
# is what makes turns expensive. It now warns the model on the way down and
# saves what it had so the run can be picked up.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("remaining,budget,expected", [
    (20, 25, ""),                 # plenty of room, say nothing
    (6, 25, ""),                  # still above the fifth
    (5, 25, "5 turns left"),      # a fifth of the budget left
    (1, 25, "1 turns left"),
    (0, 25, "LAST turn"),
    (2, 4, "2 turns left"),       # tiny budgets keep a floor of 2
])
def test_the_agent_is_told_when_it_is_running_out(remaining, budget, expected):
    assert expected in server._turn_warning(remaining, budget)


class _FakeFn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _FakeCall:
    """A tool call. Defaults to reading a file that is not there, which is the
    cheapest way to spend a turn without touching anything."""
    def __init__(self, i, name="read_file", args=None):
        self.id = f"call{i}"
        self.function = _FakeFn(name, json.dumps(args or {"path": "missing.txt"}))


class _FakeTurn:
    def __init__(self, i, name="read_file", args=None):
        self.content = None
        self.tool_calls = [_FakeCall(i, name, args)]

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": None}


class _FakeFinal:
    def __init__(self, text):
        self.content, self.tool_calls = text, []

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


class _FakeResp:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]
        self.usage = None


class _FakeClient:
    """Stands in for the OpenAI client. Keeps asking for one more tool call, so
    the only way the loop can end is by exhausting its budget — unless
    `finish_after` is set, and then it wraps up."""
    def __init__(self, finish_after=None):
        self.seen, self.calls, self.finish_after = [], 0, finish_after
        self.tool = ("read_file", {"path": "missing.txt"})
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, *, model, messages, **kw):
        self.seen.append([dict(m) for m in messages])
        self.calls += 1
        if self.finish_after is not None and self.calls >= self.finish_after:
            return _FakeResp(_FakeFinal("all done"))
        return _FakeResp(_FakeTurn(self.calls, *self.tool))


@pytest.fixture
def loop(base, tmp_path, monkeypatch):
    """Drive the agentic loop offline, with sessions kept out of the real cache."""
    monkeypatch.setattr(server, "SESSION_DIR", tmp_path / "sessions")
    client = _FakeClient()
    monkeypatch.setattr(server, "_client", lambda: client)

    def run(max_turns, tool=None):
        if tool:
            client.tool = tool
        state = {
            "base_dir": str(base),
            "messages": [{"role": "user", "content": "task"}],
            "log": [],
            "totals": {"in": 0, "out": 0, "cached": 0, "cost": 0.0},
            "turns_used": 0,
            "max_turns": max_turns,
            "turn_budget": max_turns,
        }
        return state, server._drive_agentic_loop(state)

    run.client = client
    return run


def _session_id_from(out):
    return out.split('session_id="')[1].split('"')[0]


def test_hitting_the_cap_saves_the_context_instead_of_dropping_it(loop):
    state, out = loop(3)

    assert "hit the 3-turn cap" in out
    assert "resume_delegation(session_id=" in out
    # The id it prints has to be a session that actually exists, or the advice
    # is a dead end.
    assert server._session_path(_session_id_from(out)).is_file()


def test_a_cut_off_run_carries_on_with_a_fresh_budget(loop):
    state, out = loop(3)
    sid = _session_id_from(out)

    loop.client.finish_after = loop.client.calls + 2
    resumed = server.resume_delegation(session_id=sid, result="fix the test file first")

    assert "all done" in resumed
    assert loop.client.calls > 3                      # it really kept going
    assert not server._session_path(sid).is_file()    # and the file is consumed
    # The guidance has to reach the model, not just the log.
    assert any("fix the test file first" in str(m) for m in loop.client.seen[-1])


def test_extra_turns_overrides_the_default_budget(loop):
    state, out = loop(3)
    sid = _session_id_from(out)

    server.resume_delegation(session_id=sid, result="carry on", extra_turns=1)

    assert loop.client.calls == 4  # 3 before the cap, exactly 1 granted


def test_a_session_that_is_not_waiting_for_anything_is_refused(loop, tmp_path):
    state, out = loop(3)
    sid = _session_id_from(out)
    # Strip the marker: neither a pending question nor a cut-off run.
    p = server._session_path(sid)
    saved = json.loads(p.read_text())
    saved.pop("out_of_turns")
    p.write_text(json.dumps(saved))

    assert "not waiting for an answer" in server.resume_delegation(session_id=sid, result="x")


def test_the_warning_counts_down_in_the_model_s_own_conversation(loop):
    # Budget 6 -> threshold max(2, 6//5) = 2. The six turns leave 5,4,3,2,1,0
    # behind them, so the last three are the ones that get warned.
    state, out = loop(6)

    warnings = [
        m["content"] for m in state["messages"]
        if m.get("role") == "user" and str(m.get("content", "")).startswith("TURN BUDGET:")
    ]
    assert len(warnings) == 3
    assert "2 turns left" in warnings[0]
    assert "1 turns left" in warnings[1]
    assert "LAST turn" in warnings[2]


# --------------------------------------------------------------------------
# base_dir gets Windows paths, because the project lives on C: and this server
# does not. The rejection should say so instead of just denying the path.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (r"C:\Users\josemi\Repos\app", "/mnt/c/Users/josemi/Repos/app"),
    (r"D:\code", "/mnt/d/code"),
    ("C:/Users/josemi/app", "/mnt/c/Users/josemi/app"),   # forward slashes too
    (r'"C:\quoted\path"', "/mnt/c/quoted/path"),
])
def test_a_windows_path_is_translated_for_the_hint(raw, expected):
    assert str(server._wsl_equivalent(raw)) == expected


@pytest.mark.parametrize("raw", [
    "/home/josemi/repo",
    "~/repo",
    "relative/path",
    "",
])
def test_a_posix_path_is_not_mistaken_for_a_windows_one(raw):
    assert server._wsl_equivalent(raw) is None


def test_the_rejection_points_at_the_wsl_path_when_that_one_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_wsl_equivalent", lambda raw: tmp_path)
    out = server._base_dir_error(r"C:\whatever", pathlib.Path("/nope"))
    assert "exists and looks like what you meant" in out
    assert str(tmp_path) in out


def test_the_rejection_does_not_promise_a_path_that_is_missing_too():
    out = server._base_dir_error(r"C:\definitely\not\here", pathlib.Path("/nope"))
    assert "WSL" in out
    assert "does not exist either" in out


def test_a_plain_missing_path_gets_no_wsl_noise():
    out = server._base_dir_error("/home/nobody/nothing", pathlib.Path("/home/nobody/nothing"))
    assert "WSL" not in out


# --------------------------------------------------------------------------
# Syntax checks beyond Python. `delegate_and_apply` runs nothing, so a file
# that cannot compile is reported as a success unless it is caught here.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,text", [
    ("a.json", '{"a": 1,}'),                       # trailing comma
    ("a.xml", "<a><b></a>"),                       # mismatched tag
    ("a.toml", "key = "),                          # value missing
    ("a.kt", "fun f() {\n    if (x) {\n}\n"),      # one brace short
    ("a.java", "class A { void f() { } } }"),      # one brace too many
    ("a.ts", "const a = [1, 2;"),                  # bracket never closed
    ("a.py", "def f(:\n    pass"),                 # the original check
])
def test_broken_files_are_reported(name, text):
    assert "⚠" in server._syntax_error(pathlib.Path(name), text)


@pytest.mark.parametrize("name,text", [
    ("a.json", '{"a": [1, 2], "b": {"c": null}}'),
    ("a.xml", "<a><b/></a>"),
    ("a.toml", '[tool]\nname = "x"'),
    ("a.kt", 'fun f() {\n    val s = "a } b"\n    // } in a comment\n}'),
    ("a.kt", 'val s = """\n  { unbalanced inside a raw string\n"""'),
    ("a.kt", 'val s = "${a.b()} and { }"'),        # string templates
    ("a.java", "/* } block comment */\nclass A {}"),
    ("a.go", "func f() {\n\ts := `raw } string`\n}"),
    ("a.ts", "const re = /[a-z]'/;\nfunction f() {}"),   # bails, says nothing
    ("a.md", "# a { heading with a brace"),        # not checked at all
    ("a.txt", "}}}}"),
    # Rust is not checked at all: lifetimes are indistinguishable from char
    # literals without a parser, and pretending otherwise cried wolf.
    ("a.rs", "fn f<'a>(x: &'a str) -> &'a str { x }"),
    # A double-quoted string running into a newline is a real error in most of
    # these languages, but it is also what a Rust lifetime or a PHP multi-line
    # string looks like from here, so the scanner steps back instead.
    ("a.php", '$s = "line one\nline two";'),
])
def test_good_files_are_left_alone(name, text):
    assert server._syntax_error(pathlib.Path(name), text) == ""


@pytest.mark.parametrize("text,expected", [
    ('fun f() { val s = "never closed', "unterminated string"),
    ("fun f() { /* never closed\n", "unterminated block comment"),
    ('val s = """never closed\n', "unterminated raw string"),
])
def test_unterminated_literals_are_named(text, expected):
    assert expected in server._syntax_error(pathlib.Path("a.kt"), text)


# --------------------------------------------------------------------------
# What the agent says it verified, versus what it actually ran. Those have
# differed, so the raw output of the last check comes back with the summary.
# --------------------------------------------------------------------------

def test_the_last_verification_comes_back_verbatim(loop):
    state, out = loop(2, tool=("run_bash", {"command": "echo pytest: 3 passed"}))

    assert "Last verification it ran, verbatim" in out
    assert "$ echo pytest: 3 passed" in out
    assert "pytest: 3 passed" in out


def test_commands_that_check_nothing_are_called_out(loop):
    state, out = loop(2, tool=("run_bash", {"command": "echo hello"}))

    assert "none that look like a test suite" in out


def test_a_run_that_never_executed_anything_is_flagged(loop):
    state, out = loop(2)  # only read_file calls

    assert "never ran a single command" in out


# --------------------------------------------------------------------------
# The work log records outcomes, not just attempts. Three identical commands
# say nothing; three identical commands all failing say the run is stuck.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,result,expected", [
    ("run_bash", "[exit 0]\nall good", "exit 0"),
    ("run_bash", "[exit 1]\n3 failed", "exit 1"),
    ("run_bash", "[exit -9]\nkilled", "exit -9"),          # signals, not just codes
    ("run_bash", "", "no output"),
    ("write_file", "OK: wrote a/b.py (19 lines)", "19 lines"),
    ("write_file", "OK: wrote a/b.py (1 line)", "1 lines"),
    ("read_file", "one\ntwo\nthree", "3 lines read"),
    ("list_files", "(no matches)", "no matches"),
    ("list_files", "a.py\nb.py", "2 files"),
])
def test_each_tool_call_says_what_it_did(name, result, expected):
    assert server._outcome(name, result) == expected


@pytest.mark.parametrize("result", [
    "ERROR: missing.txt does not exist",
    "REFUSED: commits, pushes and anything that publishes are done by Claude",
])
def test_a_refusal_or_error_is_carried_into_the_log(result):
    assert server._outcome("run_bash", result).startswith(result.split(":")[0])


def test_a_long_error_is_cut_down_to_a_label():
    assert len(server._outcome("read_file", "ERROR: " + "x" * 500)) <= 60


def test_the_log_shows_the_result_beside_the_attempt(loop):
    state, out = loop(2, tool=("run_bash", {"command": "exit 3"}))

    assert "run_bash(exit 3) → exit 3" in out


def test_a_stuck_run_is_visible_as_a_repeated_failure(loop):
    # The whole point: the same command failing the same way, over and over,
    # should be readable at a glance without opening anything.
    state, out = loop(3, tool=("run_bash", {"command": "false"}))

    stuck = [ln for ln in out.splitlines() if ln.endswith("run_bash(false) → exit 1")]
    assert len(stuck) == 3


def test_a_tool_with_nothing_to_report_leaves_the_line_bare():
    # A tool added later, before anyone teaches this function about it: better a
    # plain line than a dangling arrow with nothing after it.
    assert server._outcome("some_future_tool", "whatever it returned") == ""


# --------------------------------------------------------------------------
# Prices. Every cost this server reports comes from three numbers, and a
# duplicated copy of one of them is a copy that goes stale in silence.
# --------------------------------------------------------------------------

def test_the_cost_of_a_call_is_what_the_price_list_says():
    # 1M prompt tokens, none cached, plus 1M completion tokens.
    assert server._cost(1_000_000, 0, 1_000_000) == pytest.approx(0.95 + 4.00)


def test_cached_prompt_tokens_are_billed_at_the_cached_rate():
    # The same million tokens in, entirely served from the prefix cache.
    assert server._cost(1_000_000, 1_000_000, 0) == pytest.approx(0.19)


def test_nothing_used_costs_nothing():
    assert server._cost(0, 0, 0) == 0.0


def test_the_prices_appear_exactly_once_in_the_source():
    """A second copy of a price is one nobody remembers to update."""
    tree = ast.parse(pathlib.Path(server.__file__).read_text())
    prices = {server.PRICE_IN_PER_M, server.PRICE_CACHED_PER_M, server.PRICE_OUT_PER_M}

    # The module-level constants are the one place they are allowed to be.
    declared = {
        id(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
    }

    stray = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, float)
        and n.value in prices
        and id(n) not in declared
    ]
    assert not stray, f"price literals repeated outside the constants: {stray}"


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
