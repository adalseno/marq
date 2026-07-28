"""Property-based tests over the pure, input-heavy helpers.

These complement the example-based tests rather than replacing them.
Every function here is already at or near 100% *line* coverage - the gap
they close is *input* coverage: `extract_snippet` alone has three
interacting optional parameters, two recursive re-entries and a pile of
offset arithmetic, and a latent bound violation survived full line
coverage until a property was written down (see
`test_extract_snippet_body_never_exceeds_max_len`).

Scope is deliberately narrow. Property testing earns its keep on total
functions with stateable invariants; it does not earn it against Postgres
or the LLM router, where this project's real bugs have lived and where
live verification remains the tool that finds them. The one integration
property below is the exception that proves the rule: `build_ts_query`
emits SQL, so the only honest statement of its contract is "Postgres
accepts this".

Determinism: `conftest.py` loads a `derandomize=True` profile, so a given
commit always explores the same inputs. See the comment there.
"""

from collections.abc import Callable, Iterator

import psycopg
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from qmd_py.cli.snippet import extract_snippet
from qmd_py.config import get_settings
from qmd_py.search.fts import build_ts_query
from qmd_py.search.vector import CHUNK_SIZE_TOKENS, chunk_document
from qmd_py.vpath import build_virtual_path, parse_virtual_path

# Bodies that look like documents: newlines matter (every helper here is
# line-oriented), and so do the punctuation and quote characters the lex
# parser treats specially.
_BODY_ALPHABET = st.characters(
    whitelist_categories=("Ll", "Lu", "Nd", "Zs", "Po"), whitelist_characters="\n"
)
bodies = st.text(alphabet=_BODY_ALPHABET, max_size=400)
queries = st.text(alphabet=_BODY_ALPHABET, max_size=40)

# chunk_document only splits above CHUNK_SIZE_TOKENS * 3 chars, and asking
# for strings that long directly gives hypothesis a base example it cannot
# shrink (it fails the `large_base_example` health check outright).
# Repeating a short generated unit keeps the base example small while
# still reliably reaching the multi-chunk path.
_MAX_CHUNK_CHARS = CHUNK_SIZE_TOKENS * 3
long_bodies = st.builds(
    lambda unit: unit * (3 * _MAX_CHUNK_CHARS // len(unit) + 1),
    st.text(alphabet=_BODY_ALPHABET, min_size=1, max_size=40),
)
chunkable_bodies = st.one_of(bodies, long_bodies)


# =============================================================================
# extract_snippet - the densest offset arithmetic in the codebase
# =============================================================================


# The truncation branch switches form at max_len == 4 - the first budget
# the three-character ellipsis fits inside. Instrumenting the generated
# inputs showed hypothesis tries 0, 3 and 5 but never 4, and because the
# profile is derandomized that gap is stable: it would stay uncovered on
# every future run rather than being filled by chance. So the boundary is
# pinned explicitly instead of being left to the strategy.
_TRUNCATION_BOUNDARY = {"body": "alpha\nbeta gamma delta\nzeta", "query": "beta", "chunk_pos": None}


@example(**_TRUNCATION_BOUNDARY, max_len=3)
@example(**_TRUNCATION_BOUNDARY, max_len=4)
@example(**_TRUNCATION_BOUNDARY, max_len=5)
@given(
    body=bodies,
    query=queries,
    max_len=st.integers(min_value=0, max_value=200),
    chunk_pos=st.one_of(st.none(), st.integers(min_value=-5, max_value=500)),
)
def test_extract_snippet_body_never_exceeds_max_len(
    body: str, query: str, max_len: int, chunk_pos: int | None
) -> None:
    """The snippet body honours `max_len` for every `max_len`.

    This is the property that caught something: the truncation branch
    appended a three-character ellipsis *after* slicing to `max_len - 3`,
    so any `max_len <= 3` sliced from the end and then returned more than
    the caller asked for. Unreachable from marq's own call sites (they all
    pass 300 or 500), which is exactly why line coverage never saw it.

    The `@@ ... @@` header is deliberately excluded: it is metadata the
    function prepends after truncation, not part of the budget.
    """
    result = extract_snippet(body, query, max_len, chunk_pos, None, None)
    body_text = result.snippet.split("\n", 1)[1] if "\n" in result.snippet else ""
    assert len(body_text) <= max_len


@pytest.mark.parametrize("max_len", [0, 1, 2, 3, 4, 5, 6, 20])
def test_extract_snippet_truncation_is_tight_and_switches_form_at_four(max_len: int) -> None:
    """Truncation spends the whole budget, and uses the ellipsis exactly
    when it fits.

    The property above only pins the upper bound (`<= max_len`), which a
    function returning the empty string would also satisfy. This pins the
    two things that make the bound *useful*: when truncation happens the
    result is exactly `max_len` long, and the `...` marker appears for
    every budget that can hold it (>= 4) and for none that can't.

    Parametrized rather than generated on purpose - these are the specific
    values either side of the branch, and they should be checked on every
    run regardless of what the strategy happens to explore.
    """
    body = "alpha\nbeta gamma delta\nzeta"
    result = extract_snippet(body, "beta", max_len, None, None, None)
    body_text = result.snippet.split("\n", 1)[1] if "\n" in result.snippet else ""

    # The chosen body is far longer than any max_len here, so truncation
    # always fires and the bound is always reached.
    assert len(body_text) == max_len
    assert body_text.endswith("...") == (max_len >= 4)


@given(
    body=bodies,
    query=queries,
    chunk_pos=st.one_of(st.none(), st.integers(min_value=-5, max_value=500)),
    intent=st.one_of(st.none(), queries),
)
def test_extract_snippet_line_accounting_stays_consistent(
    body: str, query: str, chunk_pos: int | None, intent: str | None
) -> None:
    """The reported line numbers describe a real window into the body.

    `lines_before`/`lines_after` are rendered straight into the `@@` header
    a user reads, and they are computed from an absolute line number that
    survives a chunk-relative search plus up to two recursive re-entries.
    A negative count there would be visible nonsense; a `line` outside the
    document would send `marq get file.md:<line>` somewhere that isn't
    there.
    """
    total_lines = body.count("\n") + 1
    result = extract_snippet(body, query, 500, chunk_pos, None, intent)

    assert result.lines_before >= 0
    assert result.lines_after >= 0
    assert result.snippet_lines >= 1
    assert 1 <= result.line <= total_lines
    # The window plus what it claims to omit must add up to the document.
    assert result.lines_before + result.snippet_lines + result.lines_after == total_lines


# =============================================================================
# chunk_document - invariants its own docstring already states
# =============================================================================


@given(body=chunkable_bodies)
def test_chunk_document_chunks_are_real_slices_at_their_offsets(body: str) -> None:
    """Each `(text, offset)` pair must be exactly what `body[offset:]`
    starts with.

    The offsets are what `_pick_best_chunk` later hands to
    `extract_snippet` as `chunk_pos`, so an off-by-one here surfaces much
    later as a snippet pointing at the wrong part of the document - a
    quality bug with no exception to trace.
    """
    chunks = chunk_document(body)

    # Documented: never empty, so a document can't silently go unembedded.
    assert chunks
    for chunk_text, offset in chunks:
        assert 0 <= offset <= len(body)
        assert body[offset : offset + len(chunk_text)] == chunk_text


@given(body=chunkable_bodies)
def test_chunk_document_covers_the_whole_body(body: str) -> None:
    """Every character is inside at least one chunk.

    A gap between chunks is unretrievable text: indexed, embedded, and
    invisible to vector search for reasons nothing reports.
    """
    chunks = chunk_document(body)
    covered = 0
    for chunk_text, offset in chunks:
        assert offset <= covered, "gap between chunks"
        covered = max(covered, offset + len(chunk_text))
    assert covered == len(body)


@given(body=long_bodies)
def test_chunk_document_consecutive_chunks_overlap(body: str) -> None:
    """Consecutive chunks overlap, so a match spanning a boundary is still
    retrievable - the reason the overlap constant exists at all."""
    chunks = chunk_document(body)
    assume(len(chunks) > 1)
    for (_, first_offset), (_, second_offset) in zip(chunks, chunks[1:], strict=False):
        assert second_offset > first_offset, "chunks must advance"
        assert second_offset < first_offset + _MAX_CHUNK_CHARS, "chunks must overlap"


# =============================================================================
# vpath - a round trip with an unwritten precondition
# =============================================================================


@given(
    collection=st.text(alphabet=_BODY_ALPHABET, min_size=1, max_size=30),
    path=st.text(alphabet=_BODY_ALPHABET, max_size=60),
)
def test_virtual_path_round_trips(collection: str, path: str) -> None:
    """`parse(build(c, p)) == (c, p)`.

    Two preconditions fall out of the regex rather than being written
    down anywhere, so they are asserted as assumptions here: the
    collection segment is `[^/]+`, so it must be non-empty and slash-free
    (a slash would be read as the collection/path boundary), and `$` plus
    a non-DOTALL `(.*)` means neither half may contain a newline. Both
    hold for real collection names and real file paths.
    """
    assume("/" not in collection)
    assume("\n" not in collection and "\n" not in path)
    assume(collection.strip() == collection and collection.strip())

    parsed = parse_virtual_path(build_virtual_path(collection, path))

    assert parsed == (collection, path)


# =============================================================================
# build_ts_query - a hand-rolled parser whose output is SQL
# =============================================================================


@given(query=queries)
def test_build_ts_query_is_total(query: str) -> None:
    """The parser never raises, whatever it is handed.

    It walks a hand-managed index over the raw string with quote and `-`
    handling; unbalanced quotes and lone `-` are ordinary user input
    (CLAUDE.md documents that a stray `-` in a plain query is *not* an
    error), so the parser has to absorb them rather than throw.
    """
    result = build_ts_query(query)
    assert result is None or isinstance(result, str)


@given(query=queries)
def test_build_ts_query_never_emits_a_negative_only_expression(query: str) -> None:
    """A tsquery of nothing but negations matches everything, which is the
    opposite of what a user typing only `-foo` means. The parser returns
    None instead - the documented behaviour, asserted here for arbitrary
    input rather than the handful of examples in test_fts.py."""
    result = build_ts_query(query)
    if result is not None:
        assert not result.lstrip().startswith("!")
        assert result.strip()


@pytest.fixture(scope="module")
def tsquery_probe() -> Iterator[Callable[[str], None]]:
    """A sync, autocommitting connection for checking one tsquery string.

    Deliberately *not* the async `session` fixture, for three reasons.
    Hypothesis cannot drive an async test at all (it raises
    `InvalidArgument` rather than silently passing - checked). It also
    health-checks function-scoped fixtures, which are not reset between
    generated examples. And autocommit is what keeps shrinking honest: a
    rejected tsquery aborts its transaction, so on a shared session every
    example after the first failure would fail with "current transaction
    is aborted" instead of its own verdict, and hypothesis would shrink
    toward the wrong input.

    `to_tsquery` is a builtin - no schema, no tables, nothing marq owns is
    touched, so this needs none of the schema-per-run harness.
    """
    url = get_settings().postgres_url.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(url, autocommit=True) as conn:

        def probe(tsquery: str) -> None:
            with conn.cursor() as cur:
                cur.execute("SELECT to_tsquery('english', %s)", (tsquery,))

        yield probe


@pytest.mark.integration
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(query=queries)
def test_build_ts_query_output_is_accepted_by_postgres(
    tsquery_probe: Callable[[str], None], query: str
) -> None:
    """Whatever the parser emits, `to_tsquery` must accept it.

    This is the one property worth paying a database round trip for. The
    parser's output is interpolated into a search query, so a malformed
    expression isn't a wrong result - it's a raw `SyntaxError` from
    Postgres surfacing as a failed search. This project has already
    shipped that exact bug class once (a `ts_rank` weight array outside
    the range Postgres accepts), and no Python-side assertion substitutes
    for asking Postgres.

    The health-check suppression is safe here specifically because the
    fixture is read-only and autocommitting: there is no per-example state
    to reset.
    """
    result = build_ts_query(query)
    assume(result is not None)
    assert result is not None  # assume() filters at runtime; it doesn't narrow types

    tsquery_probe(result)
