"""Application settings, loaded from environment / .env.

qmd-py lives in its own `qmd_py` Postgres schema inside the same `qmd`
database the TS/Postgres reference implementation uses (not a separate
database - see the project plan's "Single Postgres, forever" decision).
Isolation from the TS tables (which live in `public`) comes from the
`search_path` connection option below, not from a different database.
"""

from functools import lru_cache
from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MARQ_", env_file=".env", extra="ignore")

    postgres_url: str
    """postgresql+psycopg://user:pass@host/db - no search_path needed here,
    it's appended automatically in `sqlalchemy_url`."""

    postgres_schema: str = "qmd_py"

    llm_base_url: str = "http://ubuserver.internal:8099"

    embed_model: str = "bge-m3-q8_0"
    rerank_model: str = "qwen3-reranker-0.6b-q8_0"
    # Matches the real preset id on the router (llm-stack/models/preset.ini) -
    # "qwen2.5-3b-instruct" alone (no quantization suffix) isn't a valid
    # model id there; caught live when Phase 8's query expansion first
    # exercised this setting and got a "model not found" error.
    generate_model: str = "qwen2.5-3b-instruct-q4_k_m"

    default_user_email: str = "local@marq.local"
    """Identifies the single mocked user (see auth.py). Created on first use."""

    @property
    def sqlalchemy_url(self) -> str:
        # Deliberately just our own schema, NOT `qmd_py,public`. TSVECTOR is
        # a built-in Postgres type (doesn't need `public` on the path at
        # all), and a multi-entry search_path breaks Alembic autogenerate in
        # a subtle way: Postgres's `pg_table_is_visible()` (which
        # SQLAlchemy's dialect uses for "reflect the default/implicit
        # schema") considers a table visible if ANY search_path entry
        # resolves to it - so with `qmd_py,public` on the path, the TS
        # reference implementation's tables in `public` got swept into
        # autogenerate's comparison and it proposed dropping them, even
        # though `env.py`'s `include_name` schema-level filter correctly
        # excluded `public` from the *explicit* multi-schema reflection
        # pass. That's a separate pass from the "default schema" one, and
        # only the multi-entry search_path triggers it.
        # Once Phase 5 needs pgvector's `vector` type (extension-provided,
        # unlike TSVECTOR), revisit: either re-add `public` here and solve
        # the autogenerate issue some other way (e.g. schema-qualifying the
        # vector type explicitly), or install `pgvector` into `qmd_py`
        # itself so this schema never needs `public` on its search_path.
        options = quote(f"-c search_path={self.postgres_schema}")
        separator = "&" if "?" in self.postgres_url else "?"
        return f"{self.postgres_url}{separator}options={options}"


@lru_cache
def get_settings() -> Settings:
    """Lazily constructed so importing this module doesn't require
    MARQ_POSTGRES_URL to already be set (e.g. during test collection)."""
    return Settings()  # type: ignore[call-arg]  # postgres_url comes from env
