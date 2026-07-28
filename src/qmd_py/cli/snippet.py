"""Snippet extraction for search-result output - port of the TS
reference's `extractSnippet`/`extractIntentTerms` (src/store.ts).
"""

import re
from dataclasses import dataclass

_INTENT_STOP_WORDS = frozenset(
    {
        # 2-char function words
        "am", "an", "as", "at", "be", "by", "do", "he", "if",
        "in", "is", "it", "me", "my", "no", "of", "on", "or", "so",
        "to", "up", "us", "we",
        # 3-char function words
        "all", "and", "any", "are", "but", "can", "did", "for", "get",
        "has", "her", "him", "his", "how", "its", "let", "may", "not",
        "our", "out", "the", "too", "was", "who", "why", "you",
        # 4+ char common words
        "also", "does", "find", "from", "have", "into", "more", "need",
        "show", "some", "tell", "that", "them", "this", "want", "what",
        "when", "will", "with", "your",
        # Search-context noise
        "about", "looking", "notes", "search", "where", "which",
    }
)  # fmt: skip

_PUNCT_STRIP = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def extract_intent_terms(intent: str) -> list[str]:
    terms = []
    for token in intent.lower().split():
        stripped = _PUNCT_STRIP.sub("", token)
        if len(stripped) > 1 and stripped not in _INTENT_STOP_WORDS:
            terms.append(stripped)
    return terms


INTENT_WEIGHT_SNIPPET = 0.3

# Mirrors the TS reference's char-based chunk-size constant used as the
# fallback search window when a chunk's own length isn't known.
_CHUNK_SIZE_CHARS = 900 * 3


@dataclass
class SnippetResult:
    line: int
    snippet: str
    lines_before: int
    lines_after: int
    snippet_lines: int


def extract_snippet(
    body: str,
    query: str,
    max_len: int = 500,
    chunk_pos: int | None = None,
    chunk_len: int | None = None,
    intent: str | None = None,
) -> SnippetResult:
    total_lines = body.count("\n") + 1
    search_body = body
    line_offset = 0

    if chunk_pos is not None and chunk_pos >= 0:
        search_len = chunk_len or _CHUNK_SIZE_CHARS
        context_start = max(0, chunk_pos - 100)
        context_end = min(len(body), chunk_pos + search_len + 100)
        search_body = body[context_start:context_end]
        if context_start > 0:
            line_offset = body[:context_start].count("\n")

    lines = search_body.split("\n")
    query_terms = [t for t in query.lower().split() if t]
    intent_terms = extract_intent_terms(intent) if intent else []
    best_line = 0
    best_score = -1.0

    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = 0.0
        for term in query_terms:
            if term in line_lower:
                score += 1.0
        for term in intent_terms:
            if term in line_lower:
                score += INTENT_WEIGHT_SNIPPET
        if score > best_score:
            best_score = score
            best_line = i

    if chunk_pos is not None and chunk_pos >= 0 and best_score <= 0:
        if chunk_pos == 0:
            return extract_snippet(body, query, max_len, None, None, intent)
        context_start = max(0, chunk_pos - 100)
        best_line = (
            search_body[: chunk_pos - context_start].count("\n") if chunk_pos > context_start else 0
        )

    start = max(0, best_line - 1)
    end = min(len(lines), best_line + 3)
    snippet_lines_list = lines[start:end]
    snippet_text = "\n".join(snippet_lines_list)

    if chunk_pos and chunk_pos > 0 and not snippet_text.strip():
        return extract_snippet(body, query, max_len, None, None, intent)

    if len(snippet_text) > max_len:
        # The ellipsis has to fit inside the budget too. At max_len <= 3
        # there is no room for it: `snippet_text[: max_len - 3]` slices
        # from the *end* (negative index) and then appends three more
        # characters, returning something longer than the caller asked
        # for. Unreachable from marq itself - every call site passes 300
        # or 500 - but the property test asserts the bound for any
        # max_len, so the function honours it for any max_len.
        snippet_text = (
            snippet_text[: max_len - 3] + "..." if max_len > 3 else snippet_text[:max_len]
        )

    absolute_start = line_offset + start + 1  # 1-indexed
    snippet_line_count = len(snippet_lines_list)
    lines_before = absolute_start - 1
    lines_after = total_lines - (absolute_start + snippet_line_count - 1)

    header = (
        f"@@ -{absolute_start},{snippet_line_count} @@ "
        f"({lines_before} before, {lines_after} after)"
    )
    snippet = f"{header}\n{snippet_text}"

    return SnippetResult(
        line=line_offset + best_line + 1,
        snippet=snippet,
        lines_before=lines_before,
        lines_after=lines_after,
        snippet_lines=snippet_line_count,
    )
