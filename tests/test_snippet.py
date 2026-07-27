"""Snippet extraction tests (cli/snippet.py).

Pure unit tests - no Postgres, no LLM router. This module had no direct
coverage: it was only exercised incidentally through the formatter and
CLI tests, which never reached the `chunk_pos` window logic that vector
search relies on to anchor a snippet near the actual hit rather than the
top of the document.
"""

from qmd_py.cli.snippet import (
    INTENT_WEIGHT_SNIPPET,
    SnippetResult,
    extract_intent_terms,
    extract_snippet,
)

# =============================================================================
# extract_intent_terms
# =============================================================================


def test_intent_terms_lowercase_and_strip_punctuation() -> None:
    assert extract_intent_terms("Core Web Vitals!") == ["core", "web", "vitals"]


def test_intent_terms_drop_stop_words() -> None:
    assert extract_intent_terms("what is the load time") == ["load", "time"]


def test_intent_terms_drop_single_characters() -> None:
    assert extract_intent_terms("a b performance") == ["performance"]


def test_intent_terms_of_only_stop_words_is_empty() -> None:
    assert extract_intent_terms("what are you looking for") == []


def test_intent_terms_keep_internal_punctuation() -> None:
    """Only leading/trailing punctuation is stripped, so hyphenated and
    dotted terms survive as one term."""
    assert extract_intent_terms("(state-of-the-art)") == ["state-of-the-art"]


# =============================================================================
# extract_snippet - plain (no chunk_pos)
# =============================================================================

_BODY = "alpha line\nbeta line\ngamma line\ndelta line\nepsilon line\n"


def test_snippet_centres_on_the_matching_line() -> None:
    result = extract_snippet(_BODY, "gamma")

    assert result.line == 3
    assert "gamma line" in result.snippet


def test_snippet_starts_one_line_before_the_match() -> None:
    """The window is best_line - 1 through best_line + 3, so a match on
    line 3 shows line 2 as leading context."""
    result = extract_snippet(_BODY, "gamma")

    assert "beta line" in result.snippet
    assert "alpha line" not in result.snippet


def test_snippet_header_reports_position_and_surrounding_counts() -> None:
    result = extract_snippet(_BODY, "gamma")

    assert result.snippet.startswith("@@ -2,4 @@ (1 before, 1 after)")
    assert result.lines_before == 1
    assert result.lines_after == 1
    assert result.snippet_lines == 4


def test_snippet_without_any_match_falls_back_to_the_first_line() -> None:
    result = extract_snippet(_BODY, "nothing-matches-this")

    assert result.line == 1
    assert "alpha line" in result.snippet


def test_snippet_prefers_the_line_matching_more_query_terms() -> None:
    body = "one two\nnothing\none two three\n"
    result = extract_snippet(body, "one two three")

    assert result.line == 3


def test_snippet_uses_intent_terms_to_break_a_tie() -> None:
    """Intent terms score INTENT_WEIGHT_SNIPPET each, so with no query
    term present they alone decide the line."""
    body = "plain line\nline about latency\n"
    result = extract_snippet(body, "absent-term", intent="latency")

    assert INTENT_WEIGHT_SNIPPET > 0
    assert result.line == 2


def test_snippet_query_term_outweighs_an_intent_term() -> None:
    body = "line with latency\nline with target\n"
    result = extract_snippet(body, "target", intent="latency")

    assert result.line == 2


def test_snippet_truncates_at_max_len_with_ellipsis() -> None:
    body = "\n".join("x" * 200 for _ in range(4))
    result = extract_snippet(body, "x" * 200, max_len=50)

    body_text = result.snippet.split("\n", 1)[1]
    assert len(body_text) == 50
    assert body_text.endswith("...")


def test_snippet_of_single_line_body() -> None:
    result = extract_snippet("only one line", "one")

    assert result.line == 1
    assert result.lines_before == 0
    assert result.lines_after == 0


def test_snippet_of_empty_body() -> None:
    result = extract_snippet("", "anything")

    assert isinstance(result, SnippetResult)
    assert result.line == 1


# =============================================================================
# extract_snippet - chunk_pos window (vector-search anchoring)
# =============================================================================

_LONG_BODY = "\n".join(f"line {i:03d} filler" for i in range(60)) + "\n"


def test_chunk_pos_window_offsets_line_numbers_to_the_document() -> None:
    """With a chunk far into the document, the search window starts at
    chunk_pos - 100, and reported line numbers must be absolute (offset by
    the lines skipped), not relative to the window."""
    target = _LONG_BODY.index("line 040")
    result = extract_snippet(_LONG_BODY, "line 040", chunk_pos=target)

    assert result.line == 41
    assert "line 040 filler" in result.snippet


def test_chunk_pos_window_ignores_matches_outside_it() -> None:
    """`line 002` exists, but far outside the window around chunk_pos, so
    it can't be selected."""
    target = _LONG_BODY.index("line 040")
    result = extract_snippet(_LONG_BODY, "line 002", chunk_pos=target, chunk_len=60)

    assert "line 002 filler" not in result.snippet


def test_chunk_pos_zero_without_a_match_falls_back_to_whole_body() -> None:
    """chunk_pos == 0 and nothing matched re-runs the extraction over the
    entire body rather than anchoring at the top of the chunk."""
    result = extract_snippet(_LONG_BODY, "no-such-term", chunk_pos=0)

    assert result.line == 1


def test_chunk_pos_without_a_match_anchors_at_the_chunk() -> None:
    """No query term in the window, so the snippet anchors on the chunk's
    own position instead of drifting to the window's first line."""
    target = _LONG_BODY.index("line 040")
    result = extract_snippet(_LONG_BODY, "no-such-term", chunk_pos=target)

    assert result.line > 1
    assert "line 04" in result.snippet


def test_chunk_pos_landing_in_blank_lines_falls_back_to_whole_body() -> None:
    """If anchoring at the chunk yields a whitespace-only snippet, the
    extraction restarts over the whole body so the caller never gets an
    empty excerpt. Needs a blank run long enough that the whole 4-line
    window lands inside it, past the 100-char window lead-in."""
    body = "alpha heading\n" + "\n" * 300 + "omega trailer\n"

    result = extract_snippet(body, "no-such-term", chunk_pos=200)

    assert result.snippet.split("\n", 1)[1].strip()
    assert result.line == 1


def test_negative_chunk_pos_is_treated_as_absent() -> None:
    result = extract_snippet(_BODY, "gamma", chunk_pos=-1)

    assert result.line == 3
