"""Async-command bootstrap shared by every CLI command: click callbacks
are plain sync functions, so each command's real body is an async
function this module drives to completion, opening a session and
resolving the (mocked, single) current user first.
"""

import asyncio
from collections.abc import Awaitable, Callable

from qmd_py.auth import get_current_user
from qmd_py.db.engine import get_session


async def with_session_and_user[R](
    fn: Callable[..., Awaitable[R]], *args: object, **kwargs: object
) -> R:
    async with get_session() as session:
        user = await get_current_user(session)
        await session.commit()
        return await fn(session, user, *args, **kwargs)


def run[R](fn: Callable[..., Awaitable[R]], *args: object, **kwargs: object) -> R:
    """Synchronously run an async command body, e.g.:

        run(_search_impl, query, collection=collection, format=format)

    where `_search_impl(session, user, query, *, collection, format)` is
    the actual async implementation.
    """
    return asyncio.run(with_session_and_user(fn, *args, **kwargs))
