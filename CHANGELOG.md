# Changelog

Notable changes, newest first. Dates are when the work landed on `main`.

This project has had one user and one real project so far, so "stable" here
means "its guards are tested and its documented behaviour is true", not
"battle-tested across many environments".

## 1.0.0 — 2026-08-24

First tagged release. Everything below already existed; the tag exists so there
is a version to name and to come back to.

### Added
- **Continuous integration.** Tests run on Python 3.10 – 3.13 and `ruff check`
  on every push and pull request. Nothing had ever run the suite except a human
  remembering to.
- **A pinned lint configuration** (`ruff.toml`), including `target-version`, so
  CI and a developer's machine disagree about nothing.
- **Turn-cap recovery.** A run that exhausts `max_turns` now saves its session
  like a paused one does, and `resume_delegation` continues it with a fresh
  budget (`extra_turns`). Previously the entire context was discarded and the
  next attempt re-paid for exploring the project.
- **Turn-budget warnings.** Once a fifth of the budget is left, each turn
  carries a countdown; the final turn is told to stop calling tools and
  summarise, so it stops cleanly instead of mid-edit.
- **Syntax checks beyond Python.** Exact parsers for JSON, XML and TOML; bracket
  counting for the C family (Kotlin, Java, TypeScript, Go, C#, …) with literals
  and comments skipped. Rust is deliberately excluded — a lifetime is
  indistinguishable from a char literal without a real parser.
- **Verbatim verification output.** Agentic runs echo the raw output of the last
  command that looked like a test suite or a linter, and say so explicitly when
  nothing was ever run.
- **Outcomes in the work log.** Each tool call is recorded with what came of it
  (`exit 1`, `19 lines`, `no matches`), which makes a stalled loop — the same
  command failing the same way — readable at a glance.
- **A WSL hint on `base_dir`.** A Windows path now gets its `/mnt/...`
  equivalent named, but only when that path actually exists.

### Fixed
- **Python 3.10 support, which the README had been promising and the code had
  quietly broken.** Two separate causes, both found by adding CI and pinning
  `target-version`: an unguarded `tomllib` import (3.11+), and f-strings using
  PEP 701 quote reuse and backslashes (3.12+) in the environment probe. The
  module would not even have parsed.
- Dead `_client()` call in `delegate_agentic`; the loop makes its own.

### Changed
- **Prices are declared once.** `PRICE_IN_PER_M`, `PRICE_CACHED_PER_M` and
  `PRICE_OUT_PER_M`, with a single `_cost()`. A test walks the AST and fails if
  a price is re-inlined anywhere, because a stale duplicate makes every cost
  this server reports quietly wrong.
- `subprocess.run` calls state `check=False` explicitly, and the deliberate
  blind excepts carry a `# noqa: BLE001` naming why each one is deliberate.
- README: a plain-language walkthrough of the agent loop before any API
  signature, so `max_turns` arrives with a meaning attached.

## Before 1.0.0

See `git log`. In summary, by date:

- **2026-08-21** — first working server: `delegate_to_kimi` (propose) and
  `delegate_and_apply` (write files, return a diff).
- **2026-08-22** — `delegate_agentic`, the autonomous loop with a shell;
  truncated replies refused via `finish_reason`; the environment probe;
  the requirement that delegated code be verifiable by something the model
  did not write.
- **2026-08-23** — `ask_claude` and `resume_delegation`, so the agent can
  escalate what it cannot reach; the server translated to English, prompts
  included; and a test suite (72 tests) where there had been none.
