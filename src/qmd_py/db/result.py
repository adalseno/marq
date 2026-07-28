"""Typing shim over SQLAlchemy's execute() result.

Lives in `db/` rather than in `store/` because `search/` needs it too,
and `search/` can't import from `store/` - the dependency runs the other
way (store calls `search.fts.update_document_search_vector` after every
document write).
"""

from typing import Any, cast

from sqlalchemy import CursorResult
from sqlalchemy.engine import Result


def affected_rows(result: Result[Any]) -> int:
    """How many rows a DELETE/UPDATE touched.

    `AsyncSession.execute()` is typed as returning `Result[Any]`, which
    has no `rowcount` - but a DML statement really returns a
    `CursorResult`, which does. Narrowing here (rather than an
    `attr-defined` ignore at each call site) keeps the arithmetic
    type-checked: an ignore would leave the whole expression `Any`.

    Returns:
        The row count, or 0 when the driver reports None or -1
        ("unknown"), so callers can treat it as a plain count.
    """
    rowcount = cast(CursorResult[Any], result).rowcount
    return rowcount if rowcount and rowcount > 0 else 0
