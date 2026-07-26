"""`qmd://collection/path` virtual path parsing - port of the TS
reference's `isVirtualPath`/`parseVirtualPath`/`buildVirtualPath`
(src/store.ts), minus the `?index=` query param handling, which was for
the per-project SQLite `--index <name>` concept the plan drops entirely
(it doesn't apply to a shared Postgres backend).
"""

import re

_VPATH_RE = re.compile(r"^qmd://([^/]+)/?(.*)$")


def is_virtual_path(path: str) -> bool:
    return path.startswith("qmd:") or path.startswith("//")


def parse_virtual_path(path: str) -> tuple[str, str] | None:
    normalized = path if path.startswith("qmd:") else f"qmd:{path}"
    match = _VPATH_RE.match(normalized)
    if not match:
        return None
    return match.group(1), match.group(2)


def build_virtual_path(collection_name: str, path: str) -> str:
    return f"qmd://{collection_name}/{path}"
