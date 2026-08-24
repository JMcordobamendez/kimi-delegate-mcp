#!/usr/bin/env python3
"""MCP server that lets Claude Code delegate bounded coding tasks to Kimi K2.7-code.

Two tools, with different cost/safety trade-offs:

- `delegate_to_kimi` returns the proposed code as text. Claude reviews it
  before writing anything, but then has to re-emit the whole file through
  Write/Edit — so the code is paid for twice (once as Kimi output, once as
  Claude output at 2.5x the price).

- `delegate_and_apply` writes the files itself and returns only a compact
  diff. Claude never re-emits the code, which is where the real saving is.
  Review moves from before-the-write to after-the-write, so it wants a git
  working tree underneath as the undo mechanism.
"""
import ast
import json
import os
import re
import subprocess
import time
import tomllib
import uuid
from difflib import unified_diff
from pathlib import Path
from xml.etree import ElementTree

from mcp.server.mcpserver import MCPServer
from openai import OpenAI

MODEL = "kimi-k2.7-code"
BASE_URL = "https://api.moonshot.ai/v1"

MAX_DIFF_LINES = 120

# K2.7-code always thinks and cannot be told not to; those reasoning tokens are
# billed as output AND count against this budget. Measured: 2314 of 3635
# completion tokens (64%) were reasoning on a moderate task, so a budget sized
# for the code alone truncates it. The API accepts at least 131072 here.
MAX_OUTPUT_TOKENS = 32000

# The model sometimes emits terminal colour codes around file paths, especially
# under partial mode — '\x1b[1;34mapp.py\x1b[0m' would otherwise become a real
# filename with escape bytes in it.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# The delegated model marking its own homework is the known weak spot of this
# whole design: cheap models write tests that pass without testing anything. So
# every mode asks for code that someone *else* can test independently — seams
# to inject at, no hidden state — rather than trusting the tests it wrote.
TESTABILITY_RULE = (
    "Leave the code so that someone else can test it independently: pure "
    "functions wherever possible, and side effects (network, disk, current "
    "time, randomness, subprocesses) isolated behind injectable parameters "
    "rather than buried in the logic. No hidden global state, and no "
    "dependencies that cannot be substituted."
)

# Kimi caches by *prefix*: identical leading tokens are billed at $0.19/M
# instead of $0.95/M. So everything stable must come first and the varying
# task must come last, or the prefix diverges on the first request and
# nothing is ever cached. These system prompts are byte-for-byte identical on
# every call, so they always head the cacheable prefix.
SYSTEM_PROMPT = (
    "You are a support model given bounded coding tasks. Return the complete "
    "code for every file you change or create, each in its own code block "
    "with the file path as a heading before the block. You have no way to run "
    "anything on the user's system: propose text and code only, and do not "
    "assume access to any tools.\n\n"
    + TESTABILITY_RULE
)

SYSTEM_PROMPT_APPLY = (
    "You are a support model given bounded coding tasks. Your reply is parsed "
    "automatically and written to disk, so the format is mandatory and "
    "strict.\n\n"
    "For every file you create or modify, emit exactly:\n"
    "FILE: relative/path/to/file.ext\n"
    "followed by a ``` fenced code block containing the COMPLETE, final "
    "contents of the file. Never fragments, never diffs, never '...' or any "
    "ellipsis: what you emit replaces the entire file.\n\n"
    "Paths are always relative to the working directory. Do not use absolute "
    "paths or '..'. Emit no text outside that format, except at most a single "
    "closing summary line.\n\n"
    + TESTABILITY_RULE
)

SENSITIVE_NAME_HINTS = (
    ".env",
    "credentials",
    "secret",
    "id_rsa",
    ".pem",
    ".key",
    "token",
)

# Matches: FILE: some/path.py  followed by a fenced code block.
_FILE_BLOCK_RE = re.compile(
    r"^FILE:[ \t]*(?P<path>\S.*?)[ \t]*\r?\n"
    r"```[^\n]*\r?\n"
    r"(?P<body>.*?)"
    r"^```",
    re.MULTILINE | re.DOTALL,
)

mcp = MCPServer("kimi-delegate")


def _client() -> OpenAI:
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOONSHOT_API_KEY is not set in the MCP process environment."
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def _is_sensitive(name: str) -> bool:
    return any(hint in name.lower() for hint in SENSITIVE_NAME_HINTS)


def _read_file(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if _is_sensitive(p.name):
        raise ValueError(
            f"'{path_str}' looks like a sensitive file (credentials/secrets) — "
            "it is not sent to an external provider. Pass it explicitly if you are sure."
        )
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p.read_text(errors="replace")


def _build_prompt(task: str, file_paths: list[str], extra_context: str) -> str:
    # Sorted so the same set of files always yields the same prefix, whatever
    # order the caller passed them in — an unstable order alone would defeat
    # prefix caching.
    context_blocks = []
    for fp in sorted(file_paths):
        try:
            content = _read_file(fp)
            context_blocks.append(f"### {fp}\n```\n{content}\n```")
        except Exception as e:
            context_blocks.append(f"### {fp}\n[ERROR reading the file: {e}]")

    # Stable first, varying last: file context repeats across delegations over
    # the same code, the task never does.
    parts = []
    if context_blocks:
        parts.append("Context files:\n\n" + "\n\n".join(context_blocks))
    if extra_context:
        parts.append(f"Additional context:\n{extra_context}")
    parts.append(f"Task:\n{task}")
    return "\n\n".join(parts)


def _call_kimi(system_prompt: str, prompt: str) -> tuple[str, str, str]:
    """Returns (message, usage footer, finish_reason).

    Moonshot's partial mode (prefilling an assistant turn with "partial": true)
    was tried here to force the FILE: format and skip the model's preamble. It
    does cut output tokens by ~60% — because it suppresses thinking entirely —
    but it also derails the model: it emitted empty code blocks with ANSI colour
    codes where the path should be, then asked for the path it had just been
    given. Not worth it; K2.7-code is a thinking model and works better left to
    answer normally.
    """
    client = _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    choice = response.choices[0]
    message = choice.message.content or ""

    footer = ""
    usage = response.usage
    if usage:
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        ctd = getattr(usage, "completion_tokens_details", None)
        reasoning = (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
        # Reported so cache behaviour and thinking overhead can be verified
        # rather than assumed.
        billed_in = usage.prompt_tokens - cached
        cost = (billed_in * 0.95 + cached * 0.19 + usage.completion_tokens * 4.00) / 1e6
        think = f", {reasoning} reasoning" if reasoning else ""
        footer = (
            f"_Kimi K2.7-code · {usage.prompt_tokens} tokens in "
            f"({cached} cached) / {usage.completion_tokens} tokens out"
            f"{think} · ~${cost:.5f}_"
        )
    return message, footer, choice.finish_reason or ""


# A Windows path is the one wrong answer this argument gets often enough to be
# worth naming: the project lives on C:, the editor and the build run on
# Windows, and this server does not — it runs under WSL, where "C:\..." is just
# a filename that does not exist.
_WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _wsl_equivalent(raw: str) -> Path | None:
    """The /mnt path a Windows path would mean under WSL, or None if it is not one."""
    if os.name == "nt":
        return None
    m = _WINDOWS_PATH_RE.match(raw.strip().strip('"\''))
    if not m:
        return None
    return Path(f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}")


def _base_dir_error(raw: str, resolved: Path) -> str:
    """The base_dir rejection, with the WSL translation when that is the bug."""
    plain = f"ERROR: base_dir does not exist or is not a directory: {resolved}"
    wsl = _wsl_equivalent(raw)
    if wsl is None:
        return plain
    # Only claim the translation when it is true. A hint that turns out to be
    # wrong costs more than no hint at all.
    if wsl.is_dir():
        return (
            f"ERROR: base_dir does not exist here: {raw}\n\n"
            f"That is a Windows path and this server runs inside WSL. This one "
            f"exists and looks like what you meant:\n\n    {wsl}"
        )
    return (
        f"{plain}\n\n"
        f"That looks like a Windows path, and this server runs inside WSL, so it "
        f"needs the /mnt form. {wsl} does not exist either — check the path."
    )


# Languages checked by counting brackets rather than parsing. Kotlin, Java and
# TypeScript arrive here as often as Python does, and nothing else in this
# server would notice a file that cannot compile.
_CFAMILY_SUFFIXES = {
    ".kt", ".kts", ".java", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".go", ".swift",
    ".scala", ".groovy", ".gradle", ".php",
}

# Rust is deliberately absent. A lifetime opens with the same character as a
# char literal, so `fn f<'a>(x: &'a str)` pairs the two apostrophes into a
# "literal" that swallows the '(' and reports a stray ')'. Telling the two apart
# needs a real parser, and `cargo check` is one.

_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}


def _unbalanced_brackets(text: str) -> str:
    """Walk C-family source ignoring literals, and report a bracket that cannot match.

    Deliberately not a parser. It reports only what no valid file can contain —
    a closer with nothing open, something still open at the end, an unterminated
    string or block comment — and gives up silently the moment it meets
    something it does not model, such as a regex literal. A checker that cries
    wolf is one nobody reads, so the bias is heavily towards saying nothing.
    """
    stack: list[tuple[str, int]] = []
    line, i, n = 1, 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
            continue

        if text[i:i + 2] == "//":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue

        if text[i:i + 2] == "/*":
            j = text.find("*/", i + 2)
            if j < 0:
                return f"unterminated block comment opened on line {line}"
            line += text.count("\n", i, j)
            i = j + 2
            continue

        # Raw strings (Kotlin, Scala, Groovy) hold whatever they like, braces
        # included, so they have to be skipped whole.
        if text[i:i + 3] == '"""':
            j = text.find('"""', i + 3)
            if j < 0:
                return f"unterminated raw string opened on line {line}"
            line += text.count("\n", i, j)
            i = j + 3
            continue

        if c in "\"'`":
            j, opened_at = i + 1, line
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == c:
                    break
                if text[j] == "\n" and c != "`":
                    # A bare newline inside a one-line literal means this is
                    # something we do not model (a regex literal, say). Stop
                    # rather than report a bracket count we no longer trust.
                    return ""
                j += 1
            if j >= n:
                return f"unterminated string opened on line {opened_at}"
            line += text.count("\n", i, j)
            i = j + 1
            continue

        if c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            if not stack:
                return f"stray '{c}' on line {line} with nothing open"
            opener, opened_at = stack.pop()
            if opener != _BRACKET_PAIRS[c]:
                return (
                    f"'{c}' on line {line} does not close "
                    f"'{opener}' from line {opened_at}"
                )
        i += 1

    if stack:
        opener, opened_at = stack[-1]
        return f"'{opener}' opened on line {opened_at} is never closed"
    return ""


def _syntax_error(rel: Path, text: str) -> str:
    """Report a written file that is obviously broken, before anyone trusts it.

    `delegate_and_apply` has no execution step, so nothing else would notice: a
    single stray character makes the module unimportable and the run still
    reports success. Seen for real — the model appended a lone '}' after an
    otherwise correct function.

    Where a stdlib parser exists the check is exact. Where it does not, brackets
    are counted instead, which catches the shape of that same failure without
    dragging in a compiler per language.
    """
    suffix = rel.suffix.lower()

    if suffix == ".py":
        try:
            ast.parse(text)
        except SyntaxError as e:
            return f"  ⚠ {rel}: does not parse — {e.msg} (line {e.lineno})"
        return ""

    if suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            return f"  ⚠ {rel}: does not parse — {e.msg} (line {e.lineno})"
        return ""

    if suffix in {".xml", ".svg", ".xsd", ".xsl", ".xslt", ".plist", ".pom"}:
        try:
            ElementTree.fromstring(text)
        except ElementTree.ParseError as e:
            return f"  ⚠ {rel}: does not parse — {e}"
        return ""

    if suffix == ".toml":
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            return f"  ⚠ {rel}: does not parse — {e}"
        return ""

    if suffix in _CFAMILY_SUFFIXES:
        note = _unbalanced_brackets(text)
        if note:
            return f"  ⚠ {rel}: {note}"

    return ""


def _resolve_write_path(base: Path, raw: str) -> Path:
    """Resolve a model-supplied path, refusing anything outside `base`.

    The model's output is untrusted input here: it decides these paths, so
    every one is checked against the base directory after resolution (which
    collapses '..' and follows symlinks) before anything is written.
    """
    rel = _ANSI_RE.sub("", raw).strip().strip("`\"'*").strip()
    # A path that cleans down to nothing would resolve to base_dir itself and
    # try to write over the directory. Seen for real: the model once emitted a
    # bare ANSI reset code where the path belonged.
    if not rel:
        raise ValueError(f"path was empty after cleaning: {raw!r}")
    if rel.startswith("/") or rel.startswith("~"):
        raise ValueError(f"absolute path refused: {rel}")
    target = (base / rel).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"path outside the working directory: {rel}")
    if _is_sensitive(target.name):
        raise ValueError(f"sensitive filename, not written: {rel}")
    return target


def _git_summary(base: Path) -> tuple[str, str]:
    """Return (diff --stat, untracked files) for reporting what changed."""
    try:
        diff = subprocess.run(
            ["git", "-C", str(base), "diff", "--stat"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        untracked = subprocess.run(
            ["git", "-C", str(base), "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        return diff, untracked
    except Exception:
        return "", ""


def _git_note(base: Path) -> str:
    """Warn when there is no clean git tree to undo an unwanted write."""
    try:
        inside = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return "⚠ Not a git repo: there is no easy way to undo these changes."
        dirty = subprocess.run(
            ["git", "-C", str(base), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            return (
                "⚠ The repo already had uncommitted changes before writing: "
                "Kimi's changes are now mixed in with them."
            )
        return ""
    except Exception:
        return "⚠ Could not check the git status."


@mcp.tool()
def delegate_to_kimi(task: str, file_paths: list[str] = [], extra_context: str = "") -> str:
    """Delegate a coding task to Kimi K2.7-code and return the proposed code.

    Kimi writes nothing: it returns text for you to review and apply with
    Write/Edit. Safer, but more expensive — re-emitting the code to apply it
    means paying for it twice. For bulk work where the saving matters, use
    `delegate_and_apply`.

    Do not delegate code from projects holding other people's personal data
    (GDPR).

    Args:
        task: a clear, bounded description of what Kimi should do.
        file_paths: paths (relative to the cwd, or absolute) of context files.
            Any whose name looks sensitive is refused.
        extra_context: additional free-text context.
    """
    prompt = _build_prompt(task, file_paths, extra_context)
    message, footer, finish = _call_kimi(SYSTEM_PROMPT, prompt)
    if finish == "length":
        message += (
            "\n\n⚠️ **Reply truncated by the token limit** — it is incomplete. "
            "Do not apply it as-is; split the task up and retry."
        )
    return f"{message}\n\n---\n{footer}" if footer else message


@mcp.tool()
def delegate_and_apply(
    task: str,
    base_dir: str,
    file_paths: list[str] = [],
    extra_context: str = "",
) -> str:
    """Delegate a task to Kimi and **write the files directly**, returning a diff.

    This is the cheap variant: because the code never comes back through your
    context for you to re-emit, you do not pay for it twice. In exchange,
    review moves to *after* the write — read the diff it returns and use git to
    revert if something is off. Use it on a clean git repo.

    Every write is confined to `base_dir`: absolute paths, '..' and sensitive
    filenames are refused.

    Do not delegate code from projects holding other people's personal data
    (GDPR).

    Args:
        task: a clear, bounded description of what Kimi should do.
        base_dir: the project root. This is the write boundary.
        file_paths: paths of context files to pass to Kimi.
        extra_context: additional free-text context.
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.is_dir():
        return _base_dir_error(base_dir, base)

    git_note = _git_note(base)

    prompt = _build_prompt(task, file_paths, extra_context)
    message, footer, finish = _call_kimi(SYSTEM_PROMPT_APPLY, prompt)

    # A truncated reply silently loses whichever file was mid-generation: the
    # regex needs a closing fence, so the last block just doesn't match and the
    # run reports partial work as success. Refuse to write anything instead.
    if finish == "length":
        return (
            "ABORTED: Kimi's reply was cut off by the token limit "
            f"(finish_reason='length'), so it is incomplete. **Nothing was "
            "written** — applying it would leave half-finished files or "
            "silently lose work. Split the task into smaller parts and "
            "retry.\n\n"
            f"---\n{footer}"
        )

    blocks = list(_FILE_BLOCK_RE.finditer(message))
    if not blocks:
        # Nothing parseable — hand back the raw text so the work isn't lost.
        return (
            "No 'FILE:' block was found in the reply; nothing was written. "
            "Raw reply from Kimi:\n\n"
            f"{message}\n\n---\n{footer}"
        )

    summary: list[str] = []
    diff_lines: list[str] = []
    errors: list[str] = []
    broken: list[str] = []

    for m in blocks:
        try:
            target = _resolve_write_path(base, m.group("path"))
        except ValueError as e:
            errors.append(f"  ✗ {e}")
            continue

        new_text = m.group("body")
        rel = target.relative_to(base)
        existed = target.is_file()
        old_text = target.read_text(errors="replace") if existed else ""

        if existed and old_text == new_text:
            summary.append(f"  = {rel} (unchanged)")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_text)

        bad = _syntax_error(rel, new_text)
        if bad:
            broken.append(bad)

        if existed:
            old_lines = old_text.splitlines(keepends=True)
            new_lines = new_text.splitlines(keepends=True)
            d = list(unified_diff(old_lines, new_lines, str(rel), str(rel), n=2))
            added = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
            removed = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
            summary.append(f"  M {rel} (+{added} -{removed})")
            diff_lines.extend(x.rstrip("\n") for x in d)
        else:
            n = len(new_text.splitlines())
            summary.append(f"  A {rel} (new, {n} lines)")

    out = [f"Written in {base}:", *summary]
    if errors:
        out += ["", "Refused:", *errors]
    if broken:
        out += [
            "",
            "⚠️ **Written but broken** — review before trusting it:",
            *broken,
        ]
    if git_note:
        out += ["", git_note]

    if diff_lines:
        shown = diff_lines[:MAX_DIFF_LINES]
        out += ["", "--- diff ---", *shown]
        if len(diff_lines) > MAX_DIFF_LINES:
            out.append(
                f"... (diff clipped: {len(diff_lines) - MAX_DIFF_LINES} more lines; "
                "use git diff to see it in full)"
            )

    if footer:
        out += ["", "---", footer]
    return "\n".join(out)


SYSTEM_PROMPT_AGENTIC = (
    "You are a programming agent working inside a project. You have tools to "
    "read, write and list files, and to run shell commands. The working "
    "directory is already set to the project root: always use relative "
    "paths.\n\n"
    "How to work: explore whatever you need, make the changes, and **verify by "
    "running the tests or whatever command applies**. If something fails, fix "
    "it and run it again. Do not call the task finished without having "
    "actually checked it.\n\n"
    "Do not commit, push, or publish anything: Claude handles that with the "
    "user's approval. Do not use git to discard or rewrite changes either. "
    "Leave your work in the working tree as it is.\n\n"
    "If you need something beyond your reach — flashing or resetting hardware, "
    "reading a serial port, someone physically looking at an LED or a display, "
    "touching files outside the project, using credentials, making a commit — "
    "do NOT invent it and do not give up: use `ask_claude` to request it. Your "
    "session pauses and continues with the answer, losing no context. Ask for "
    "one thing at a time and be specific: the exact command or the exact "
    "observation you need, and what you expect to see.\n\n"
    + TESTABILITY_RULE
    + "\n\nOn top of that, leave the test infrastructure set up and working "
    "even if the task did not ask for it: the runner configured and, if "
    "needed, a conftest.py with fixtures for anything external (network, disk, "
    "clock). The goal is that whoever comes next can write a test of their own "
    "and run it without setting anything up.\n\n"
    "Your final summary must include, without fail:\n"
    "1. The EXACT command to run the tests, exactly as it works in this "
    "project (including the interpreter path if you used a venv).\n"
    "2. Which fixtures or injection points you left available.\n"
    "3. What stayed hard to test, and why.\n\n"
    "When you are finished and have verified it, reply with that summary in "
    "plain text and without calling any more tools."
)

AGENTIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a project file. Path relative to the root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or replace) a file. Path relative to the root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List project files matching a glob, e.g. '**/*.py'.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_claude",
            "description": (
                "Ask Claude to do something you cannot: flash or reset "
                "hardware, read a serial port, physically look at an LED or a "
                "display, touch files outside the project, use credentials, or "
                "make a commit. Your session pauses and continues with the "
                "answer, losing no context. Ask for ONE thing at a time and be "
                "very specific: give the exact command or the exact "
                "observation you need, and what you expect to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {"request": {"type": "string"}},
                "required": ["request"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a shell command with the working directory set to the "
                "project root. Use it above all to run the tests."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

MAX_TOOL_OUTPUT = 4000
BASH_TIMEOUT = 120

# Paused agentic loops live here between the question and the answer. Outside
# the project so a suspended run never shows up in the user's git status.
SESSION_DIR = Path.home() / ".cache" / "kimi-delegate" / "sessions"
SESSION_TTL_HOURS = 24
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _session_path(session_id: str) -> Path:
    # The id comes back through an argument, so it is untrusted: only accept
    # the exact shape we generate, never a path fragment.
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(f"malformed session_id: {session_id!r}")
    return SESSION_DIR / f"{session_id}.json"


def _sweep_old_sessions() -> None:
    cutoff = time.time() - SESSION_TTL_HOURS * 3600
    try:
        for f in SESSION_DIR.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError:
        pass


def _save_session(state: dict) -> str:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    _sweep_old_sessions()
    session_id = uuid.uuid4().hex
    _session_path(session_id).write_text(json.dumps(state))
    return session_id


def _load_session(session_id: str) -> dict:
    p = _session_path(session_id)
    if not p.is_file():
        raise FileNotFoundError(
            f"No paused delegation with id {session_id}. "
            f"It may have expired (sessions last {SESSION_TTL_HOURS} h)."
        )
    return json.loads(p.read_text())

# Commits, pushes and anything that publishes stay with Claude, done with the
# user's approval — never the delegated agent. Also refused: the git commands
# that would destroy the working tree this design relies on as its undo path.
# Read-only git (status, diff, log, ls-files) stays available.
#
# This is a workflow guard over a cooperative model, not a security boundary:
# run_bash takes a shell string, so a determined model could word its way past
# it. It exists to stop accidents, not attacks.
_FORBIDDEN_CMD_RE = re.compile(
    r"\bgit\s+(commit|push|tag|reset|rebase|merge|stash|clean|checkout|restore)\b"
    r"|\bgh\s+"
    r"|\b(npm|pnpm|yarn)\s+publish\b"
    r"|\btwine\s+upload\b"
    r"|\b(poetry|cargo)\s+publish\b",
    re.IGNORECASE,
)


def _clip(text: str, keep_tail: bool = False) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    if keep_tail:
        # Test runners and compilers put the verdict at the *end*. Keeping only
        # the head would cut off the one line the agent actually needs.
        head = MAX_TOOL_OUTPUT // 2
        tail = MAX_TOOL_OUTPUT - head
        cut = len(text) - MAX_TOOL_OUTPUT
        return f"{text[:head]}\n... [{cut} chars clipped from the middle] ...\n{text[-tail:]}"
    return text[:MAX_TOOL_OUTPUT] + f"\n... [clipped, {len(text) - MAX_TOOL_OUTPUT} more chars]"


def _probe_environment(base: Path) -> str:
    """Tell the agent what it is working with before it starts guessing.

    Turns are the expensive unit in a loop: each one costs its own output
    (reasoning included) and then rides along in every later request. In one
    run, four of eleven turns went on discovering that `python` was not on
    PATH, that pytest was missing, and building a virtualenv. Handing that over
    up front buys back those turns.
    """
    def sh(cmd: str) -> str:
        try:
            r = subprocess.run(
                cmd, shell=True, cwd=str(base),
                capture_output=True, text=True, timeout=20,
            )
            out = (r.stdout or r.stderr).strip()
            return out.splitlines()[0] if out else "no"
        except Exception:
            return "could not check"

    return (
        "Environment already checked (do not spend turns rediscovering it):\n"
        f"- interpreter: {sh('python3 --version')}\n"
        f"- pytest already installed: {sh('python3 -m pytest --version 2>&1 | head -1')}\n"
        f"- existing venv in the project: {sh('ls -d venv .venv 2>/dev/null | head -1')}\n"
        f"- project files: {sh('ls requirements*.txt pyproject.toml setup.py package.json Makefile 2>/dev/null | tr \"\\n\" \" \"')}\n"
        f"- tests already present: {sh('ls test_*.py *_test.py tests 2>/dev/null | head -5 | tr \"\\n\" \" \"')}"
    )


def _run_agentic_tool(base: Path, name: str, args: dict) -> str:
    """Execute one tool call. Never raises: the model needs to see failures."""
    try:
        if name == "read_file":
            p = _resolve_write_path(base, args["path"])
            if not p.is_file():
                return f"ERROR: {args['path']} does not exist"
            return _clip(p.read_text(errors="replace"))

        if name == "write_file":
            p = _resolve_write_path(base, args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"OK: wrote {p.relative_to(base)} ({len(args['content'].splitlines())} lines)"

        if name == "list_files":
            pattern = args.get("pattern", "**/*")
            hits = [
                str(f.relative_to(base))
                for f in sorted(base.glob(pattern))
                if f.is_file() and ".git/" not in str(f)
            ]
            return _clip("\n".join(hits[:200]) or "(no matches)")

        if name == "run_bash":
            if _FORBIDDEN_CMD_RE.search(args["command"]):
                return (
                    "REFUSED: commits, pushes and anything that publishes are "
                    "done by Claude with the user's approval, not by you. You "
                    "also cannot use git to discard or rewrite changes. Leave "
                    "the work in the tree and describe it in your final summary."
                )
            r = subprocess.run(
                args["command"], shell=True, cwd=str(base),
                capture_output=True, text=True, timeout=BASH_TIMEOUT,
            )
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
            return _clip(f"[exit {r.returncode}]\n{out}".strip(), keep_tail=True)

        return f"ERROR: unknown tool {name}"
    except subprocess.TimeoutExpired:
        return f"ERROR: the command exceeded {BASH_TIMEOUT}s and was aborted"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@mcp.tool()
def delegate_agentic(
    task: str,
    base_dir: str,
    extra_context: str = "",
    max_turns: int = 25,
) -> str:
    """Turn Kimi loose as an **autonomous agent** inside a project, with a shell.

    Unlike the other two tools, Kimi does not give an answer and stop here: it
    explores the project, edits, **runs the tests, sees them fail and corrects
    itself**, in a loop, until it is done. This is what makes best use of
    K2.7-code, which is tuned for agentic tool use.

    Requires a clean git repo: git is the undo mechanism, and when it finishes
    the `git diff --stat` of whatever it touched is returned.

    ⚠️ Kimi runs real shell commands. The working directory is pinned to
    `base_dir`, but a command can leave it with an absolute path: the
    confinement is a convenience, not a cage. Everything it reads is sent to
    Moonshot. Do not use on projects holding third-party data (GDPR).

    Args:
        task: the goal, with its "done" criterion (e.g. "until pytest passes").
        base_dir: the project root. Must be a git repo with no pending changes.
        extra_context: additional free-text context.
        max_turns: iteration cap, so a loop cannot run away. Hitting it is
            not the end of the road: the context is saved and
            `resume_delegation` picks the run up with a fresh budget.
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.is_dir():
        return _base_dir_error(base_dir, base)

    note = _git_note(base)
    if note:
        return (
            f"ABORTED before starting: {note}\n\n"
            "Agentic mode runs commands and edits files with no review in "
            "between, so it requires a clean git repo to make undo possible. "
            "Commit or discard whatever is pending and retry."
        )

    # Stable-first ordering again: the environment block is the same across
    # delegations into the same project, so it sits ahead of the varying task
    # and stays inside the cacheable prefix.
    env_block = _probe_environment(base)
    context = f"{env_block}\n\n{extra_context}" if extra_context else env_block

    client = _client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC},
        {"role": "user", "content": _build_prompt(task, [], context)},
    ]

    state = {
        "base_dir": str(base),
        "messages": messages,
        "log": [],
        "totals": {"in": 0, "out": 0, "cached": 0, "cost": 0.0},
        "turns_used": 0,
        "max_turns": max_turns,
        # `max_turns` grows when a cut-off run is resumed; this one does not, so
        # the warning window stays the same size on every stretch.
        "turn_budget": max_turns,
    }
    return _drive_agentic_loop(state)


# The loop used to just stop when it ran out of turns, mid-whatever it was
# doing. Observed failure: a run that hit the cap having written a test file
# that did not compile — worse than one that stopped cleanly, because the tree
# is left broken and the caller has to finish it by hand. Telling the model how
# much rope is left lets it land the plane.
# What counts as the agent checking its own work. The point is not to be
# exhaustive: it is to find the one command whose raw output settles whether the
# run succeeded, so the summary is not the only evidence on offer.
_VERIFY_CMD_RE = re.compile(
    r"\b(pytest|unittest\b|tox\b|nox\b"
    r"|npm\s+(run\s+)?test|yarn\s+test|pnpm\s+test|jest\b|vitest\b"
    r"|go\s+test|cargo\s+(test|clippy|check)|mvn\b|gradlew?\b|sbt\b"
    r"|make\s+(test|check)|ctest\b|rspec\b|phpunit\b"
    r"|ruff\b|flake8\b|mypy\b|pyright\b|eslint\b|tsc\b|clippy\b)"
)


def _outcome(name: str, result: str) -> str:
    """One short label for what a tool call actually did.

    The work log used to record only the attempt, and an attempt reads exactly
    the same whether the agent is converging or going round in circles. Three
    identical `run_bash` lines say nothing; three identical lines all ending
    `exit 1` say the run is stuck. The result is already in hand at the point
    the line is written — it just has to be said out loud.

    Written by the server from what happened, never by the agent about itself:
    a summary from a stuck model is a claim made by the least reliable witness
    at its least reliable moment.
    """
    first = result.splitlines()[0] if result else ""
    if first.startswith(("ERROR:", "REFUSED:", "ABORTED")):
        return first[:60]

    if name == "run_bash":
        m = re.match(r"\[exit (-?\d+)\]", result)
        return f"exit {m.group(1)}" if m else "no output"

    if name == "write_file":
        m = re.search(r"\((\d+) lines?\)", first)
        return f"{m.group(1)} lines" if m else "written"

    if name == "read_file":
        return f"{len(result.splitlines())} lines read"

    if name == "list_files":
        if result.strip() == "(no matches)":
            return "no matches"
        return f"{len(result.splitlines())} files"

    return ""


def _turn_warning(remaining: int, budget: int) -> str:
    """The nudge to inject before a turn, or "" while there is room to work."""
    if remaining == 0:
        return (
            "TURN BUDGET: this is your LAST turn. Do not start anything new and "
            "do not call any more tools. Reply with a summary of what is done, "
            "what is left, and anything you left half-finished."
        )
    if remaining <= max(2, budget // 5):
        return (
            f"TURN BUDGET: {remaining} turns left before you are cut off. Stop "
            "opening new fronts. Get what you have already written into a state "
            "that compiles and passes, then summarise."
        )
    return ""


def _drive_agentic_loop(state: dict) -> str:
    """Run the loop until it finishes, runs out of turns, or asks for help.

    Split out of `delegate_agentic` so a paused run can be picked up again by
    `resume_delegation` with its context intact.
    """
    base = Path(state["base_dir"])
    messages = state["messages"]
    log = state["log"]
    totals = state["totals"]
    client = _client()
    final = ""
    cap_session_id = ""

    while state["turns_used"] < state["max_turns"]:
        state["turns_used"] += 1
        turn = state["turns_used"]

        # Turns left AFTER this one. Inside the danger zone the count goes into
        # every turn: the pressure is meant to rise, and a single warning eight
        # turns back is one the model has stopped acting on.
        remaining = state["max_turns"] - state["turns_used"]
        warning = _turn_warning(remaining, state.get("turn_budget", state["max_turns"]))
        if warning:
            messages.append({"role": "user", "content": warning})

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=AGENTIC_TOOLS,
            tool_choice="auto",
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
        msg = resp.choices[0].message

        u = resp.usage
        if u:
            ptd = getattr(u, "prompt_tokens_details", None)
            cached = (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
            totals["cost"] += (
                (u.prompt_tokens - cached) * 0.95 + cached * 0.19 + u.completion_tokens * 4.00
            ) / 1e6
            totals["in"] += u.prompt_tokens
            totals["out"] += u.completion_tokens
            totals["cached"] += cached

        # The assistant turn must go back verbatim — including reasoning_content,
        # which Moonshot rejects the next request without. Dropping it is the
        # exact bug that makes some thinking models unusable in tool loops.
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            final = msg.content or ""
            break

        pending: tuple[str, str] | None = None
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args, result = {}, f"ERROR: invalid JSON arguments: {e}"
            else:
                if name == "ask_claude":
                    if pending is None:
                        # Leave this one unanswered and suspend below; its tool
                        # result is what resume_delegation will supply.
                        pending = (tc.id, args.get("request", ""))
                        log.append(f"  {turn:>2}. ask_claude → paused")
                        continue
                    result = (
                        "ERROR: you can only ask for one thing at a time. "
                        "Ask this again once the previous one is answered."
                    )
                else:
                    result = _run_agentic_tool(base, name, args)
                    # Keep the raw output of whatever last checked the work. The
                    # agent's prose about it has been wrong before; this has not.
                    if name == "run_bash" and _VERIFY_CMD_RE.search(args.get("command", "")):
                        state["last_verification"] = {
                            "command": args["command"],
                            "output": result,
                        }

            detail = args.get("command") or args.get("path") or args.get("pattern") or ""
            outcome = _outcome(name, result)
            line = f"  {turn:>2}. {name}({detail[:70]})"
            log.append(f"{line} → {outcome}" if outcome else line)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if pending is not None:
            call_id, request = pending
            state["pending_tool_call_id"] = call_id
            session_id = _save_session(state)
            diff, untracked = _git_summary(base)
            # Without the work so far, whoever answers is doing it blind — and a
            # mistaken answer sends the agent chasing a bug that isn't there.
            out = [
                "⏸️  PAUSED — Kimi needs something it cannot do itself:",
                "",
                f"    {request}",
                "",
                f"What it has done so far ({len(log)} calls):",
                *log,
            ]
            if diff:
                out += ["", "git diff --stat:", diff]
            if untracked:
                out += ["", "New untracked files:", untracked]
            out += [
                "",
                "Check that before answering: if what it is asking for is "
                "already solved, or if your answer contradicts what it has "
                "written, tell it so instead of letting it guess.",
                "",
                "Do what it asks (ask the user first if it involves hardware, "
                "credentials, or anything irreversible) and return the result "
                "with:",
                "",
                f'    resume_delegation(session_id="{session_id}", result="...")',
                "",
                f"Its context is saved. ~${totals['cost']:.5f} spent so far.",
            ]
            return "\n".join(out)
    else:
        # Save it here too, not only on a pause. Without this the whole context
        # is thrown away and the next attempt re-pays the exploration, which is
        # exactly what makes the turns expensive in the first place.
        state["out_of_turns"] = True
        cap_session_id = _save_session(state)
        final = f"(unfinished: hit the {state['max_turns']}-turn cap)"

    total_in, total_out = totals["in"], totals["out"]
    total_cached, total_cost = totals["cached"], totals["cost"]

    diff, untracked = _git_summary(base)

    out = [f"Kimi worked in {base} — {len(log)} tool calls:", *log]
    if final:
        out += ["", "Kimi's summary:", final]
    if diff:
        out += ["", "git diff --stat:", diff]
    if untracked:
        out += ["", "New untracked files:", untracked]

    # "All 14 tests pass" is a claim; this is the evidence. They have differed.
    checked = state.get("last_verification")
    if checked:
        out += [
            "",
            "Last verification it ran, verbatim — not its summary of it:",
            f"    $ {checked['command']}",
            *[f"    {ln}" for ln in checked["output"].splitlines()],
        ]
    elif any("run_bash(" in entry for entry in log):
        out += [
            "",
            "⚠️  It ran commands, but none that look like a test suite or a "
            "linter. Nothing here has been checked by running it.",
        ]
    else:
        out += [
            "",
            "⚠️  It never ran a single command. Agentic mode exists to close "
            "that loop, so treat this as unverified.",
        ]

    if cap_session_id:
        budget = state.get("turn_budget", state["max_turns"])
        out += [
            "",
            "⏱️  It ran out of turns, so the work above may be half-finished — "
            "read the diff before trusting it. Its context is saved, so carrying "
            "on is far cheaper than starting over:",
            "",
            f'    resume_delegation(session_id="{cap_session_id}", '
            'result="<what to finish first>")',
            "",
            f"Pass `extra_turns=N` to grant more than the {budget} it started with.",
        ]
    out += [
        "",
        "---",
        # Cache hits matter most here: every turn resends the whole history, so
        # the repeated prefix is the bulk of what is billed.
        f"_Kimi K2.7-code · {total_in} tokens in ({total_cached} cached, "
        f"{100 * total_cached // max(total_in, 1)}%) / {total_out} out "
        f"across {len(log)} calls · ~${total_cost:.5f}_",
    ]
    return "\n".join(out)


@mcp.tool()
def resume_delegation(session_id: str, result: str, extra_turns: int = 0) -> str:
    """Pick a delegation back up — after it asked for something, or ran out of turns.

    Kimi pauses when it needs something beyond its reach — flashing an ESP32,
    reading a serial port, looking at an LED, touching something outside the
    project. Do what it asked (asking the user first if it involves hardware,
    credentials, or anything irreversible) and pass back what you observed.

    Be literal: paste the real command output, or describe exactly what is
    visible. If you could not do it, say so anyway and explain why — that lets
    Kimi try another route instead of waiting.

    A run cut off by `max_turns` resumes through here too. There is no question
    to answer in that case, so `result` is guidance instead: what to finish
    first, what to leave alone. Look at the diff before writing it — a cut-off
    run can have left something half-written.

    Args:
        session_id: the id the pause, or the turn-cap message, returned.
        result: what you observed, verbatim. Also the place to say it failed.
            For a cut-off run, the guidance to carry on with.
        extra_turns: how many turns to add. 0 gives a cut-off run the same
            budget it started with, and leaves a paused one untouched.
    """
    try:
        state = _load_session(session_id)
    except (ValueError, FileNotFoundError) as e:
        return f"ERROR: {e}"

    call_id = state.pop("pending_tool_call_id", None)
    if call_id:
        state["messages"].append(
            {"role": "tool", "tool_call_id": call_id, "content": result}
        )
        if extra_turns > 0:
            state["max_turns"] += extra_turns
    elif state.pop("out_of_turns", False):
        # A cut-off run has no unanswered tool call, and no turns left to spend:
        # granting the budget IS the resumption, and `result` steers it.
        budget = state.get("turn_budget", state["max_turns"])
        state["max_turns"] += extra_turns if extra_turns > 0 else budget
        state["messages"].append({
            "role": "user",
            "content": (
                f"You were cut off by the turn cap. You now have "
                f"{state['max_turns'] - state['turns_used']} more turns. Check "
                f"what you left half-finished before carrying on.\n\n{result}"
            ),
        })
    else:
        return "ERROR: that session is not waiting for an answer."

    # Drop the saved copy now: from here the run either finishes or suspends
    # again under a fresh id, and a stale file would invite a double resume.
    _session_path(session_id).unlink(missing_ok=True)
    return _drive_agentic_loop(state)


if __name__ == "__main__":
    mcp.run()
