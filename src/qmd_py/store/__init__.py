"""Service/CRUD layer - the single facade the CLI, MCP server, and any
future REST API all call through (fixes the TS reference implementation's
inconsistency #3, where the SDK's `searchLex`/`searchVector` silently
skipped behavior the CLI's `search`/`vsearch` commands did: one code path
per operation here, not two that can drift apart).

Every function that touches a specific collection's data resolves the
`Collection` row and calls `can_access()` (see auth.py) before proceeding -
mocked to always allow today, but the choke point is real from day one.

This was one 1000-line module; it is now a package split by
responsibility. The whole public API is re-exported here, so
`from qmd_py.store import ...` keeps working exactly as before and no
caller needed changing. Submodules are layered to avoid cycles:

    _common  -> (nothing in-package)
    cleanup  -> _common
    documents-> _common
    collection, context, indexing, retrieval -> the above
"""

from qmd_py.store._common import (
    CollectionNotFoundError,
    PermissionDeniedError,
    add_line_numbers,
    extract_title,
    hash_content,
    utcnow,
)
from qmd_py.store.cleanup import (
    cleanup_orphaned_content,
    delete_inactive_documents,
    delete_llm_cache,
)
from qmd_py.store.collection import (
    CollectionListRow,
    RemoveCollectionResult,
    add_collection,
    get_collection,
    list_collections,
    remove_collection,
    rename_collection,
    set_include_by_default,
    set_update_command,
)
from qmd_py.store.context import (
    CollectionMissingContext,
    ContextRow,
    add_context,
    context_check,
    get_global_context,
    list_contexts,
    remove_context,
    set_global_context,
)
from qmd_py.store.documents import (
    deactivate_document,
    find_active_document,
    find_document_by_path,
    get_active_document_paths,
    insert_content,
    insert_document,
    update_document,
)
from qmd_py.store.indexing import ReindexResult, reindex_collection
from qmd_py.store.retrieval import (
    DEFAULT_MULTI_GET_MAX_BYTES,
    CollectionStatus,
    DocumentDetail,
    DocumentNotFound,
    FileRow,
    GlobMatch,
    MultiGetFile,
    StatusInfo,
    find_document,
    get_status,
    list_files,
    match_files_by_glob,
    multi_get,
)

__all__ = [
    "DEFAULT_MULTI_GET_MAX_BYTES",
    "CollectionListRow",
    "CollectionMissingContext",
    "CollectionNotFoundError",
    "CollectionStatus",
    "ContextRow",
    "DocumentDetail",
    "DocumentNotFound",
    "FileRow",
    "GlobMatch",
    "MultiGetFile",
    "PermissionDeniedError",
    "ReindexResult",
    "RemoveCollectionResult",
    "StatusInfo",
    "add_collection",
    "add_context",
    "add_line_numbers",
    "cleanup_orphaned_content",
    "context_check",
    "deactivate_document",
    "delete_inactive_documents",
    "delete_llm_cache",
    "extract_title",
    "find_active_document",
    "find_document",
    "find_document_by_path",
    "get_active_document_paths",
    "get_collection",
    "get_global_context",
    "get_status",
    "hash_content",
    "insert_content",
    "insert_document",
    "list_collections",
    "list_contexts",
    "list_files",
    "match_files_by_glob",
    "multi_get",
    "reindex_collection",
    "remove_collection",
    "remove_context",
    "rename_collection",
    "set_global_context",
    "set_include_by_default",
    "set_update_command",
    "update_document",
    "utcnow",
]
