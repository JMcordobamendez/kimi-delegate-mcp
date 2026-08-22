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
import json
import os
import re
import subprocess
from difflib import unified_diff
from pathlib import Path

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
    "Deja el código preparado para que otra persona pueda testearlo por su "
    "cuenta: funciones puras siempre que se pueda, y los efectos secundarios "
    "(red, disco, hora actual, aleatoriedad, subprocesos) aislados detrás de "
    "parámetros inyectables en lugar de incrustados en la lógica. Nada de "
    "estado global oculto ni de dependencias imposibles de sustituir."
)

# Kimi caches by *prefix*: identical leading tokens are billed at $0.19/M
# instead of $0.95/M. So everything stable must come first and the varying
# task must come last, or the prefix diverges on the first request and
# nothing is ever cached. These system prompts are byte-for-byte identical on
# every call, so they always head the cacheable prefix.
SYSTEM_PROMPT = (
    "Eres un modelo de apoyo al que se le delegan tareas de código acotadas. "
    "Devuelve el código completo de cada fichero que cambies o crees, cada "
    "uno en su propio bloque de código con la ruta como cabecera antes del "
    "bloque. No hay forma de que ejecutes nada en el sistema del usuario: "
    "limítate a proponer texto/código, no asumas acceso a herramientas.\n\n"
    + TESTABILITY_RULE
)

SYSTEM_PROMPT_APPLY = (
    "Eres un modelo de apoyo al que se le delegan tareas de código acotadas. "
    "Tu respuesta se parsea automáticamente y se escribe a disco, así que el "
    "formato es obligatorio y estricto.\n\n"
    "Para cada fichero que crees o modifiques emite exactamente:\n"
    "FILE: ruta/relativa/del/fichero.ext\n"
    "seguido de un bloque de código cercado con ``` que contenga el contenido "
    "COMPLETO y final del fichero. Nunca fragmentos, nunca diffs, nunca '...' "
    "ni elipsis: lo que emitas sustituye al fichero entero.\n\n"
    "Las rutas son siempre relativas al directorio de trabajo. No uses rutas "
    "absolutas ni '..'. No emitas texto fuera de ese formato, salvo como mucho "
    "una línea final de resumen.\n\n"
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
            "MOONSHOT_API_KEY no está definida en el entorno del proceso MCP."
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
            f"'{path_str}' parece un fichero sensible (credenciales/secretos) — "
            "no se envía a un proveedor externo. Pásalo explícitamente si estás seguro."
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
            context_blocks.append(f"### {fp}\n[ERROR leyendo el fichero: {e}]")

    # Stable first, varying last: file context repeats across delegations over
    # the same code, the task never does.
    parts = []
    if context_blocks:
        parts.append("Ficheros de contexto:\n\n" + "\n\n".join(context_blocks))
    if extra_context:
        parts.append(f"Contexto adicional:\n{extra_context}")
    parts.append(f"Tarea:\n{task}")
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
        think = f", {reasoning} de razonamiento" if reasoning else ""
        footer = (
            f"_Kimi K2.7-code · {usage.prompt_tokens} tokens in "
            f"({cached} cacheados) / {usage.completion_tokens} tokens out"
            f"{think} · ~${cost:.5f}_"
        )
    return message, footer, choice.finish_reason or ""


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
        raise ValueError(f"ruta vacía tras limpiarla: {raw!r}")
    if rel.startswith("/") or rel.startswith("~"):
        raise ValueError(f"ruta absoluta rechazada: {rel}")
    target = (base / rel).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"ruta fuera del directorio de trabajo: {rel}")
    if _is_sensitive(target.name):
        raise ValueError(f"nombre de fichero sensible, no se escribe: {rel}")
    return target


def _git_note(base: Path) -> str:
    """Warn when there is no clean git tree to undo an unwanted write."""
    try:
        inside = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return "⚠ No es un repo git: no hay forma sencilla de deshacer estos cambios."
        dirty = subprocess.run(
            ["git", "-C", str(base), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            return (
                "⚠ El repo ya tenía cambios sin commitear antes de escribir: "
                "los cambios de Kimi quedan mezclados con ellos."
            )
        return ""
    except Exception:
        return "⚠ No se pudo comprobar el estado de git."


@mcp.tool()
def delegate_to_kimi(task: str, file_paths: list[str] = [], extra_context: str = "") -> str:
    """Delega una tarea de código a Kimi K2.7-code y devuelve el código propuesto.

    Kimi NO escribe nada: devuelve texto que tú revisas y aplicas con
    Write/Edit. Más seguro, pero más caro — al reemitir el código para
    aplicarlo lo pagas dos veces. Para trabajo voluminoso donde el ahorro
    importe, usa `delegate_and_apply`.

    No delegar código de proyectos con datos personales de terceros (RGPD).

    Args:
        task: descripción clara y acotada de lo que debe hacer Kimi.
        file_paths: rutas (relativas al cwd, o absolutas) de ficheros de
            contexto. Los que tengan nombre de pinta sensible se rechazan.
        extra_context: contexto adicional en texto libre.
    """
    prompt = _build_prompt(task, file_paths, extra_context)
    message, footer, finish = _call_kimi(SYSTEM_PROMPT, prompt)
    if finish == "length":
        message += (
            "\n\n⚠️ **Respuesta truncada por límite de tokens** — está incompleta. "
            "No la apliques tal cual; divide la tarea y reintenta."
        )
    return f"{message}\n\n---\n{footer}" if footer else message


@mcp.tool()
def delegate_and_apply(
    task: str,
    base_dir: str,
    file_paths: list[str] = [],
    extra_context: str = "",
) -> str:
    """Delega una tarea a Kimi y **escribe los ficheros directamente**, devolviendo un diff.

    Ésta es la variante barata: como el código no vuelve por tu contexto para
    que lo reemitas, no lo pagas dos veces. A cambio la revisión pasa a ser
    *posterior* a la escritura — revisa el diff que devuelve y usa git para
    revertir si algo no cuadra. Úsala en un repo git limpio.

    Toda escritura queda confinada a `base_dir`: se rechazan rutas absolutas,
    '..' y nombres de fichero sensibles.

    No delegar código de proyectos con datos personales de terceros (RGPD).

    Args:
        task: descripción clara y acotada de lo que debe hacer Kimi.
        base_dir: directorio raíz del proyecto. Es la frontera de escritura.
        file_paths: rutas de ficheros de contexto a pasarle a Kimi.
        extra_context: contexto adicional en texto libre.
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.is_dir():
        return f"ERROR: base_dir no existe o no es un directorio: {base}"

    git_note = _git_note(base)

    prompt = _build_prompt(task, file_paths, extra_context)
    message, footer, finish = _call_kimi(SYSTEM_PROMPT_APPLY, prompt)

    # A truncated reply silently loses whichever file was mid-generation: the
    # regex needs a closing fence, so the last block just doesn't match and the
    # run reports partial work as success. Refuse to write anything instead.
    if finish == "length":
        return (
            "ABORTADO: la respuesta de Kimi se cortó por límite de tokens "
            f"(finish_reason='length'), así que está incompleta. **No se ha escrito "
            "nada** — aplicarla dejaría ficheros a medias o trabajo perdido en "
            "silencio. Divide la tarea en partes más pequeñas y reintenta.\n\n"
            f"---\n{footer}"
        )

    blocks = list(_FILE_BLOCK_RE.finditer(message))
    if not blocks:
        # Nothing parseable — hand back the raw text so the work isn't lost.
        return (
            "No se encontró ningún bloque 'FILE:' en la respuesta; no se ha "
            "escrito nada. Respuesta cruda de Kimi:\n\n"
            f"{message}\n\n---\n{footer}"
        )

    summary: list[str] = []
    diff_lines: list[str] = []
    errors: list[str] = []

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
            summary.append(f"  = {rel} (sin cambios)")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_text)

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
            summary.append(f"  A {rel} (nuevo, {n} líneas)")

    out = [f"Escrito en {base}:", *summary]
    if errors:
        out += ["", "Rechazado:", *errors]
    if git_note:
        out += ["", git_note]

    if diff_lines:
        shown = diff_lines[:MAX_DIFF_LINES]
        out += ["", "--- diff ---", *shown]
        if len(diff_lines) > MAX_DIFF_LINES:
            out.append(
                f"... (diff recortado: {len(diff_lines) - MAX_DIFF_LINES} líneas más; "
                "usa git diff para verlo entero)"
            )

    if footer:
        out += ["", "---", footer]
    return "\n".join(out)


SYSTEM_PROMPT_AGENTIC = (
    "Eres un agente de programación trabajando dentro de un proyecto. Tienes "
    "herramientas para leer, escribir y listar ficheros, y para ejecutar "
    "comandos de shell. El directorio de trabajo ya está fijado en la raíz del "
    "proyecto: usa siempre rutas relativas.\n\n"
    "Método de trabajo: explora lo que necesites, haz los cambios, y "
    "**verifica ejecutando los tests o el comando que corresponda**. Si algo "
    "falla, corrígelo y vuelve a ejecutarlo. No des la tarea por terminada sin "
    "haberla comprobado de verdad.\n\n"
    "No hagas commits ni push, ni publiques nada: de eso se encarga Claude con "
    "el visto bueno del usuario. Tampoco uses git para descartar o reescribir "
    "cambios. Deja tu trabajo en el árbol de trabajo tal cual.\n\n"
    + TESTABILITY_RULE
    + "\n\nAdemás, deja la infraestructura de tests montada y funcionando "
    "aunque la tarea no la pidiera: el runner configurado y, si hace falta, un "
    "conftest.py con fixtures para lo externo (red, disco, reloj). La idea es "
    "que quien venga detrás pueda escribir un test suyo y ejecutarlo sin "
    "montar nada.\n\n"
    "Tu resumen final debe incluir, sí o sí:\n"
    "1. El comando EXACTO para ejecutar los tests, tal y como funciona en este "
    "proyecto (con la ruta del intérprete si has usado un venv).\n"
    "2. Qué fixtures o puntos de inyección has dejado disponibles.\n"
    "3. Qué ha quedado difícil de testear y por qué.\n\n"
    "Cuando hayas terminado y verificado, responde con ese resumen en texto "
    "plano y sin llamar a más herramientas."
)

AGENTIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee un fichero del proyecto. Ruta relativa a la raíz.",
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
            "description": "Escribe (crea o reemplaza) un fichero. Ruta relativa a la raíz.",
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
            "description": "Lista ficheros del proyecto que casan con un glob, p.ej. '**/*.py'.",
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
            "name": "run_bash",
            "description": (
                "Ejecuta un comando de shell con el directorio de trabajo en la "
                "raíz del proyecto. Úsalo sobre todo para ejecutar los tests."
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


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f"\n... [recortado, {len(text) - MAX_TOOL_OUTPUT} chars más]"


def _run_agentic_tool(base: Path, name: str, args: dict) -> str:
    """Execute one tool call. Never raises: the model needs to see failures."""
    try:
        if name == "read_file":
            p = _resolve_write_path(base, args["path"])
            if not p.is_file():
                return f"ERROR: no existe {args['path']}"
            return _clip(p.read_text(errors="replace"))

        if name == "write_file":
            p = _resolve_write_path(base, args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"OK: escrito {p.relative_to(base)} ({len(args['content'].splitlines())} líneas)"

        if name == "list_files":
            pattern = args.get("pattern", "**/*")
            hits = [
                str(f.relative_to(base))
                for f in sorted(base.glob(pattern))
                if f.is_file() and ".git/" not in str(f)
            ]
            return _clip("\n".join(hits[:200]) or "(sin resultados)")

        if name == "run_bash":
            if _FORBIDDEN_CMD_RE.search(args["command"]):
                return (
                    "RECHAZADO: los commits, los push y cualquier publicación los "
                    "hace Claude con el visto bueno del usuario, no tú. Tampoco "
                    "puedes usar git para descartar o reescribir cambios. Deja el "
                    "trabajo en el árbol y descríbelo en tu resumen final."
                )
            r = subprocess.run(
                args["command"], shell=True, cwd=str(base),
                capture_output=True, text=True, timeout=BASH_TIMEOUT,
            )
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
            return _clip(f"[exit {r.returncode}]\n{out}".strip())

        return f"ERROR: herramienta desconocida {name}"
    except subprocess.TimeoutExpired:
        return f"ERROR: el comando excedió {BASH_TIMEOUT}s y se abortó"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@mcp.tool()
def delegate_agentic(
    task: str,
    base_dir: str,
    extra_context: str = "",
    max_turns: int = 25,
) -> str:
    """Lanza a Kimi como **agente autónomo** dentro de un proyecto, con shell.

    A diferencia de las otras dos herramientas, aquí Kimi no da una respuesta y
    para: explora el proyecto, edita, **ejecuta los tests, ve si fallan y se
    corrige**, en bucle, hasta terminar. Es lo que mejor aprovecha a K2.7-code,
    que está afinado para uso agéntico de herramientas.

    Requiere un repo git limpio: git es el mecanismo de deshacer, y al terminar
    se devuelve el `git diff --stat` de lo que haya tocado.

    ⚠️ Kimi ejecuta comandos de shell reales. El directorio de trabajo se fija
    a `base_dir`, pero un comando puede salirse de ahí con rutas absolutas: el
    confinamiento es de conveniencia, no una jaula. Todo lo que lea se envía a
    Moonshot. No usar en proyectos con datos de terceros (RGPD).

    Args:
        task: el objetivo, con su criterio de "terminado" (p.ej. "que pase pytest").
        base_dir: raíz del proyecto. Debe ser un repo git sin cambios pendientes.
        extra_context: contexto adicional en texto libre.
        max_turns: tope de iteraciones, para que un bucle no se desboque.
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.is_dir():
        return f"ERROR: base_dir no existe o no es un directorio: {base}"

    note = _git_note(base)
    if note:
        return (
            f"ABORTADO antes de empezar: {note}\n\n"
            "El modo agéntico ejecuta comandos y edita ficheros sin revisión "
            "intermedia, así que exige un repo git limpio para poder deshacer. "
            "Commitea o descarta lo que tengas pendiente y reintenta."
        )

    client = _client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC},
        {"role": "user", "content": _build_prompt(task, [], extra_context)},
    ]

    total_cost = 0.0
    total_in = total_out = 0
    log: list[str] = []
    final = ""

    for turn in range(1, max_turns + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=AGENTIC_TOOLS,
            tool_choice="auto",
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
        choice = resp.choices[0]
        msg = choice.message

        u = resp.usage
        if u:
            ptd = getattr(u, "prompt_tokens_details", None)
            cached = (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
            total_cost += (
                (u.prompt_tokens - cached) * 0.95 + cached * 0.19 + u.completion_tokens * 4.00
            ) / 1e6
            total_in += u.prompt_tokens
            total_out += u.completion_tokens

        # The assistant turn must go back verbatim — including reasoning_content,
        # which Moonshot rejects the next request without. Dropping it is the
        # exact bug that makes some thinking models unusable in tool loops.
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            final = msg.content or ""
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                result = f"ERROR: argumentos JSON inválidos: {e}"
                args = {}
            else:
                result = _run_agentic_tool(base, name, args)

            detail = args.get("command") or args.get("path") or args.get("pattern") or ""
            log.append(f"  {turn:>2}. {name}({detail[:70]})")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
    else:
        final = f"(sin terminar: alcanzado el tope de {max_turns} turnos)"

    try:
        diff = subprocess.run(
            ["git", "-C", str(base), "diff", "--stat"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        untracked = subprocess.run(
            ["git", "-C", str(base), "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:
        diff = untracked = ""

    out = [f"Kimi trabajó en {base} — {len(log)} llamadas a herramientas:", *log]
    if final:
        out += ["", "Resumen de Kimi:", final]
    if diff:
        out += ["", "git diff --stat:", diff]
    if untracked:
        out += ["", "Ficheros nuevos sin trackear:", untracked]
    out += [
        "",
        "---",
        f"_Kimi K2.7-code · {total_in} tokens in / {total_out} out en "
        f"{len(log)} llamadas · ~${total_cost:.5f}_",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run()
