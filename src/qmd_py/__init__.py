"""marq - centralized markdown/code search over Postgres/pgvector.

The distribution is `qmd-py` and the import package is `qmd_py`, but the
CLI and MCP surface are branded `marq`; the package name is an internal
detail no user sees. See CLAUDE.md for why renaming it is out of scope.

Nothing is re-exported here on purpose - import from the submodule that
owns the thing (`qmd_py.store`, `qmd_py.search.hybrid`, ...).
"""
