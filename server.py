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
    "limítate a proponer texto/código, no asumas acceso a herramientas."
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
    "una línea final de resumen."
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


def _call_kimi(system_prompt: str, prompt: str) -> tuple[str, str]:
    """Returns (message, usage footer)."""
    client = _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=8000,
    )
    message = response.choices[0].message.content or ""

    footer = ""
    usage = response.usage
    if usage:
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        # Reported so cache behaviour can be verified rather than assumed.
        billed_in = usage.prompt_tokens - cached
        cost = (billed_in * 0.95 + cached * 0.19 + usage.completion_tokens * 4.00) / 1e6
        footer = (
            f"_Kimi K2.7-code · {usage.prompt_tokens} tokens in "
            f"({cached} cacheados) / {usage.completion_tokens} tokens out "
            f"· ~${cost:.5f}_"
        )
    return message, footer


def _resolve_write_path(base: Path, raw: str) -> Path:
    """Resolve a model-supplied path, refusing anything outside `base`.

    The model's output is untrusted input here: it decides these paths, so
    every one is checked against the base directory after resolution (which
    collapses '..' and follows symlinks) before anything is written.
    """
    rel = raw.strip().strip("`\"'")
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
    message, footer = _call_kimi(SYSTEM_PROMPT, prompt)
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
    message, footer = _call_kimi(SYSTEM_PROMPT_APPLY, prompt)

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


if __name__ == "__main__":
    mcp.run()
