"""Full-text search: `tsvector`/`tsquery` machinery, ported from the TS
reference's `buildTsQuery`/`searchFTS`/`updateDocumentSearchVector`
(src/store.ts) - see the qmd-py plan's "Algorithmic details" section for
why the weight scaling and dotted-token handling are shaped this way.

Deliberately does not import from `store.py`: collection/ACL resolution is
duplicated locally (a few lines) rather than reused, so `store.py` can
import this module (to call `update_document_search_vector` after every
document write) without a circular import.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import ColumnElement, func, literal_column, select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser, can_access
from qmd_py.db.models import Collection, CollectionContext, Content, Document, User

# =============================================================================
# CJK handling
# =============================================================================

# Approximates the TS reference's \p{Script=Han/Hiragana/Katakana/Hangul} via
# explicit codepoint ranges (Python's stdlib `re` has no \p{Script=...}
# support) - covers the common BMP blocks, not every rare CJK extension.
_CJK_RANGES = (
    "぀-ゟ"  # Hiragana
    "゠-ヿ"  # Katakana
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # CJK Compatibility Ideographs
    "ᄀ-ᇿ"  # Hangul Jamo
    "가-힣"  # Hangul Syllables
)
_CJK_RUN_PATTERN = re.compile(f"[{_CJK_RANGES}]+")
_CJK_CHAR_PATTERN = re.compile(f"[{_CJK_RANGES}]")


def normalize_cjk_for_fts(text: str) -> str:
    """Postgres's built-in text search configs don't segment CJK runs into
    words (same gap FTS5's unicode61 tokenizer has) - space out every
    character in a CJK run so it can be treated as an adjacency phrase."""
    return _CJK_RUN_PATTERN.sub(lambda m: " " + " ".join(m.group(0)) + " ", text)


def contains_cjk(text: str) -> bool:
    return _CJK_CHAR_PATTERN.search(text) is not None


# =============================================================================
# Term sanitization / token classification
# =============================================================================


def _is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("L", "N")


def sanitize_fts_term(term: str) -> str:
    return "".join(ch for ch in term if ch in ("'", "_") or _is_word_char(ch)).lower()


def is_hyphenated_token(token: str) -> bool:
    """True for compound words like multi-agent, DEC-0054, gpt-4: starts and
    ends with a letter/number, contains at least one internal hyphen, and
    every character is a letter/number/apostrophe/hyphen."""
    if "-" not in token or not token:
        return False
    if not _is_word_char(token[0]) or not _is_word_char(token[-1]):
        return False
    return all(_is_word_char(ch) or ch in ("'", "-") for ch in token)


def is_dotted_token(token: str) -> bool:
    """True for version-like strings (2026.4.10, 3.14.0): splitting on dots
    yields >= 2 non-empty word/number/underscore parts."""
    parts = token.split(".")
    if len(parts) < 2:
        return False
    return all(p and all(_is_word_char(ch) or ch == "_" for ch in p) for p in parts)


def _to_phrase(words: list[str]) -> str | None:
    if not words:
        return None
    return f"({' <-> '.join(words)})" if len(words) > 1 else words[0]


def build_ts_query(query: str) -> str | None:
    """Parse lex query syntax into a Postgres `to_tsquery` expression.

    Supports the same surface syntax as the TS reference's FTS5-era parser:
    - Quoted phrases: "exact phrase" -> word <-> word (adjacency)
    - Negation: -term or -"phrase" -> tsquery's unary `!`
    - Hyphenated tokens: multi-agent, DEC-0054 -> phrase (not prefix) match
    - Dotted tokens: 2026.4.10 -> the whole dotted string, prefix-matched
      (Postgres's tokenizer keeps these as one lexeme, unlike SQLite FTS5,
      so - unlike the TS version - the parts are NOT split and ANDed)
    - Plain terms: term -> term:* (prefix match)

    Negative-only queries return None (no results), matching the original's
    restriction even though tsquery's `!` is technically usable standalone.
    """
    positive: list[str] = []
    negative: list[str] = []

    s = query.strip()
    i = 0
    n = len(s)

    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break

        negated = s[i] == "-"
        if negated:
            i += 1

        if i < n and s[i] == '"':
            start = i + 1
            i += 1
            while i < n and s[i] != '"':
                i += 1
            phrase = s[start:i].strip()
            i += 1  # skip closing quote
            if phrase:
                words = [
                    w
                    for t in normalize_cjk_for_fts(phrase).split()
                    if (w := sanitize_fts_term(t))
                ]
                phrase_query = _to_phrase(words)
                if phrase_query:
                    (negative if negated else positive).append(phrase_query)
        else:
            start = i
            while i < n and not s[i].isspace() and s[i] != '"':
                i += 1
            term = s[start:i]

            if is_hyphenated_token(term):
                words = [sanitize_fts_term(t) for t in term.split("-") if sanitize_fts_term(t)]
                phrase_query = _to_phrase(words)
                if phrase_query:
                    (negative if negated else positive).append(phrase_query)
            elif is_dotted_token(term):
                sanitized = "".join(
                    ch for ch in term.lower() if _is_word_char(ch) or ch in (".", "_")
                )
                if sanitized:
                    (negative if negated else positive).append(f"{sanitized}:*")
            elif contains_cjk(term):
                words = [
                    w
                    for t in normalize_cjk_for_fts(term).split()
                    if (w := sanitize_fts_term(t))
                ]
                phrase_query = _to_phrase(words)
                if phrase_query:
                    (negative if negated else positive).append(phrase_query)
            else:
                sanitized = sanitize_fts_term(term)
                if sanitized:
                    (negative if negated else positive).append(f"{sanitized}:*")

    if not positive:
        return None

    result = " & ".join(positive)
    for neg in negative:
        result = f"{result} & !{neg}"
    return result


def validate_semantic_query(query: str) -> str | None:
    if re.search(r'(^|\s)-[\w"]', query):
        return "Negation (-term) is not supported in vec/hyde queries. Use lex for exclusions."
    return None


def validate_lex_query(query: str) -> str | None:
    if re.search(r"[\r\n]", query):
        return (
            "Lex queries must be a single line. Remove newline characters or split into "
            "separate lex: lines."
        )
    if query.count('"') % 2 == 1:
        return 'Lex query has an unmatched double quote ("). Add the closing quote or remove it.'
    return None


# =============================================================================
# search_vector maintenance
# =============================================================================


async def update_document_search_vector(session: AsyncSession, document_id: int) -> None:
    """Recompute a document's `search_vector` from its current title,
    display path, and body. Not a Postgres `GENERATED` column (the body
    lives in `Content.doc`, joined via hash - a generated column's
    expression can't reach across tables) - called wherever a document's
    title/path/body changes (see store.py's insert_document/update_document).

    Weights mirror the original SQLite FTS5 `bm25(documents_fts, 1.5, 4.0,
    1.0)` weighting (filepath, title, body): title matters most, filepath
    next, body is the baseline - setweight()'s A/B/C zones encode that
    order; the actual ratio is applied at query time via ts_rank's weights
    array (see FTS_RANK_WEIGHTS below).
    """
    row = (
        await session.execute(
            select(
                col(Document.title), col(Document.path), col(Content.doc), col(Collection.name)
            )
            .join(Content, col(Content.hash) == col(Document.hash))
            .join(Collection, col(Collection.id) == col(Document.collection_id))
            .where(col(Document.id) == document_id, col(Document.active))
        )
    ).one_or_none()

    if row is None:
        await session.execute(
            sa_update(Document).where(col(Document.id) == document_id).values(search_vector=None)
        )
        return

    title, path, body, collection_name = row
    # setweight()'s zone argument is Postgres's single-byte "char" type, not
    # varchar/text - passed as a bound VARCHAR parameter, no setweight
    # overload matches. Embed it as a SQL literal instead.
    title_tsv = func.setweight(
        func.to_tsvector("english", normalize_cjk_for_fts(title)), literal_column("'A'")
    )
    path_tsv = func.setweight(
        func.to_tsvector("english", normalize_cjk_for_fts(f"{collection_name}/{path}")),
        literal_column("'B'"),
    )
    body_tsv = func.setweight(
        func.to_tsvector("english", normalize_cjk_for_fts(body)), literal_column("'C'")
    )

    await session.execute(
        sa_update(Document)
        .where(col(Document.id) == document_id)
        .values(search_vector=title_tsv.op("||")(path_tsv).op("||")(body_tsv))
    )


# =============================================================================
# search_fts
# =============================================================================


def get_docid(hash_: str) -> str:
    return hash_[:6]


async def get_context_for_path(
    session: AsyncSession, user: CurrentUser, collection_id: int, path: str
) -> str | None:
    """Hierarchical context: the user's global context (`context add /`)
    first, then all matching per-path CollectionContext rows for this
    collection, shortest/most-general prefix first - port of the TS
    reference's `getContextForPath`."""
    contexts: list[str] = []

    global_context = (
        await session.execute(select(col(User.global_context)).where(col(User.id) == user.id))
    ).scalar_one_or_none()
    if global_context:
        contexts.append(global_context)

    rows = (
        await session.execute(
            select(col(CollectionContext.path_prefix), col(CollectionContext.context)).where(
                col(CollectionContext.collection_id) == collection_id
            )
        )
    ).all()

    normalized_path = path if path.startswith("/") else f"/{path}"
    matching = []
    for prefix, context in rows:
        normalized_prefix = prefix if prefix.startswith("/") else f"/{prefix}"
        if normalized_path.startswith(normalized_prefix):
            matching.append((normalized_prefix, context))
    matching.sort(key=lambda pc: len(pc[0]))
    contexts.extend(context for _, context in matching)

    return "\n\n".join(contexts) if contexts else None


@dataclass
class SearchResult:
    filepath: str
    display_path: str
    title: str
    hash: str
    docid: str
    collection_name: str
    modified_at: datetime
    body_length: int
    body: str
    context: str | None
    score: float
    source: str = "fts"


# ts_rank's weights array is {D, C, B, A} (lowest to highest zone), each
# entry must be in [0, 1] (Postgres raises "weight out of range" otherwise).
# Scaled (/4.0) to stay proportional to the original bm25 weighting: title=
# A=1.0 (matters most), filepath=B=0.375, body=C=0.25 (baseline). D unused.
FTS_RANK_WEIGHTS = "{0.025, 0.25, 0.375, 1.0}"


async def _resolve_collection_ids(
    session: AsyncSession, user: CurrentUser, collection_name: str | None
) -> list[int]:
    result = await session.execute(select(Collection).order_by(col(Collection.name)))
    collections = [c for c in result.scalars() if await can_access(user, c, "read")]
    if collection_name is not None:
        collections = [c for c in collections if c.name == collection_name]
    return [c.id for c in collections]


async def search_fts(
    session: AsyncSession,
    user: CurrentUser,
    query: str,
    limit: int = 20,
    collection_name: str | None = None,
) -> list[SearchResult]:
    ts_query = build_ts_query(query)
    if ts_query is None:
        return []

    collection_ids = await _resolve_collection_ids(session, user, collection_name)
    if not collection_ids:
        return []

    tsq_expr = func.to_tsquery("english", ts_query)
    weights: ColumnElement[Any] = literal_column(f"'{FTS_RANK_WEIGHTS}'::float4[]")
    rank_expr = func.ts_rank(weights, col(Document.search_vector), tsq_expr, 32)

    stmt = (
        select(
            col(Document.collection_id),
            col(Document.path),
            col(Document.title),
            col(Document.hash),
            col(Document.modified_at),
            col(Content.doc).label("body"),
            rank_expr.label("rank_score"),
        )
        .select_from(Document)
        .join(Content, col(Content.hash) == col(Document.hash))
        .where(
            col(Document.active),
            col(Document.search_vector).op("@@")(tsq_expr),
            col(Document.collection_id).in_(collection_ids),
        )
        .order_by(rank_expr.desc())
        .limit(limit)
    )

    rows = (await session.execute(stmt)).all()
    if not rows:
        return []

    names_result = await session.execute(
        select(col(Collection.id), col(Collection.name)).where(
            col(Collection.id).in_({row.collection_id for row in rows})
        )
    )
    collection_names: dict[int, str] = {row.id: row.name for row in names_result}

    results = []
    for row in rows:
        name = collection_names[row.collection_id]
        context = await get_context_for_path(session, user, row.collection_id, row.path)
        results.append(
            SearchResult(
                filepath=f"qmd://{name}/{row.path}",
                display_path=f"{name}/{row.path}",
                title=row.title,
                hash=row.hash,
                docid=get_docid(row.hash),
                collection_name=name,
                modified_at=row.modified_at,
                body_length=len(row.body),
                body=row.body,
                context=context,
                score=row.rank_score,
                source="fts",
            )
        )
    return results
