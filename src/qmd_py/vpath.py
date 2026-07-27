"""`marq://collection/path` virtual path parsing - port of the TS
reference qmd's `isVirtualPath`/`parseVirtualPath`/`buildVirtualPath`
(src/store.ts), minus the `?index=` query param handling, which was for
the per-project SQLite `--index <name>` concept the plan drops entirely
(it doesn't apply to a shared Postgres backend).
"""

import re

_VPATH_RE = re.compile(r"^marq://([^/]+)/?(.*)$")


def is_virtual_path(path: str) -> bool:
    """Whether `path` looks like a virtual path rather than a plain one.

    Accepts the bare `//collection/path` form as well as `marq://...`,
    since shells and editors routinely strip the scheme.

    Returns:
        True for anything starting `marq:` or `//`. A syntactic check
        only - it neither parses the rest nor checks the collection
        exists.
    """
    return path.startswith("marq:") or path.startswith("//")


def parse_virtual_path(path: str) -> tuple[str, str] | None:
    """Split a virtual path into its collection and document path.

    Args:
        path: `marq://collection/path`, or the scheme-less `//collection/path`.

    Returns:
        A `(collection_name, path)` pair, or None if it doesn't parse.
        The path half is `""` for a bare collection reference like
        `marq://notes/`, so callers should expect an empty second element
        rather than None.
    """
    normalized = path if path.startswith("marq:") else f"marq:{path}"
    match = _VPATH_RE.match(normalized)
    if not match:
        return None
    return match.group(1), match.group(2)


def build_virtual_path(collection_name: str, path: str) -> str:
    """Join a collection and document path into a `marq://` URI.

    No escaping is applied - see `mcp/server.py`'s `_encode_qmd_path()`
    for the percent-encoded form the MCP resource layer needs.
    """
    return f"marq://{collection_name}/{path}"
