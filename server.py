#!/usr/bin/env python3
"""MCP server that lets Claude Code delegate bounded coding tasks to Kimi K2.7-code.

Kimi never touches this machine directly: it only returns text (proposed
code). Claude decides whether to apply it with its own Write/Edit tools
after reviewing the output. This keeps the review gate that the
orchestration research (vault: "Orquestar modelos baratos desde Claude Code")
identified as the correct boundary for Vía C.
"""
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from openai import OpenAI

MODEL = "kimi-k2.7-code"
BASE_URL = "https://api.moonshot.ai/v1"

# Kimi caches by *prefix*: identical leading tokens are billed at $0.19/M
# instead of $0.95/M. So everything stable must come first and the varying
# task must come last, or the prefix diverges on the first request and
# nothing is ever cached. This system prompt is byte-for-byte identical on
# every call, so it always heads the cacheable prefix.
SYSTEM_PROMPT = (
    "Eres un modelo de apoyo al que se le delegan tareas de código acotadas. "
    "Devuelve el código completo de cada fichero que cambies o crees, cada "
    "uno en su propio bloque de código con la ruta como cabecera antes del "
    "bloque. No hay forma de que ejecutes nada en el sistema del usuario: "
    "limítate a proponer texto/código, no asumas acceso a herramientas."
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

mcp = MCPServer("kimi-delegate")


def _client() -> OpenAI:
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOONSHOT_API_KEY no está definida en el entorno del proceso MCP."
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def _read_file(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if any(hint in p.name.lower() for hint in SENSITIVE_NAME_HINTS):
        raise ValueError(
            f"'{path_str}' parece un fichero sensible (credenciales/secretos) — "
            "no se envía a un proveedor externo. Pásalo explícitamente si estás seguro."
        )
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p.read_text(errors="replace")


@mcp.tool()
def delegate_to_kimi(task: str, file_paths: list[str] = [], extra_context: str = "") -> str:
    """Delega una tarea de código acotada a Kimi K2.7-code (Moonshot AI).

    Úsalo para trabajo repetitivo o de bajo riesgo: picar código a partir de
    una especificación clara, escribir tests, refactors mecánicos. Kimi NO
    ejecuta nada en este ordenador — solo devuelve texto (código propuesto).
    Revisa siempre el resultado antes de aplicarlo con Write/Edit.

    No delegar código de proyectos con datos personales de terceros (ver
    [[Automatización Facturas Inmobiliaria]] en el vault, excluida por RGPD).

    Args:
        task: descripción clara y acotada de lo que debe hacer Kimi.
        file_paths: rutas (relativas al cwd, o absolutas) de ficheros de
            contexto a incluir. Ficheros con nombre de pinta sensible
            (.env, credentials, *.pem...) se rechazan automáticamente.
        extra_context: contexto adicional en texto libre (specs, convenciones
            de estilo, fragmentos relevantes que no valga la pena leer de fichero).
    """
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

    # Stable first, varying last (see SYSTEM_PROMPT): file context repeats
    # across delegations over the same code, the task never does.
    prompt_parts = []
    if context_blocks:
        prompt_parts.append("Ficheros de contexto:\n\n" + "\n\n".join(context_blocks))
    if extra_context:
        prompt_parts.append(f"Contexto adicional:\n{extra_context}")
    prompt_parts.append(f"Tarea:\n{task}")
    prompt = "\n\n".join(prompt_parts)

    client = _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=8000,
    )
    message = response.choices[0].message.content or ""

    usage = response.usage
    if usage:
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        # Reported so cache behaviour can be verified rather than assumed.
        billed_in = usage.prompt_tokens - cached
        cost = (billed_in * 0.95 + cached * 0.19 + usage.completion_tokens * 4.00) / 1e6
        message += (
            f"\n\n---\n_Kimi K2.7-code · {usage.prompt_tokens} tokens in "
            f"({cached} cacheados) / {usage.completion_tokens} tokens out "
            f"· ~${cost:.5f}_"
        )
    return message


if __name__ == "__main__":
    mcp.run()
