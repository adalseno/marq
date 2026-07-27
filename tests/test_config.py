"""Settings tests - specifically `sqlalchemy_url`'s query-string handling.

Pure unit tests: `Settings` is constructed directly with explicit values
rather than read from the environment, so nothing here touches Postgres,
the router, or the developer's own `.env`.

`config.py` reported 100% *line* coverage without any of this, which was
misleading: the separator below is a one-line conditional whose `&` arm
had never executed, because every URL the suite ever built (and the one
in `.env`) is bare. A DSN carrying `?sslmode=require` is entirely normal
for a managed or TLS-only Postgres, and getting the separator wrong there
produces a second `?` - a malformed URL that fails at connect time, well
away from this code.
"""

from qmd_py.config import Settings

_SEARCH_PATH = "options=-c%20search_path%3Dmyschema"


def _settings(url: str) -> Settings:
    return Settings(postgres_url=url, postgres_schema="myschema")


def test_bare_url_gets_a_question_mark_separator() -> None:
    url = _settings("postgresql+psycopg://u:p@h/db").sqlalchemy_url

    assert url == f"postgresql+psycopg://u:p@h/db?{_SEARCH_PATH}"


def test_url_with_an_existing_query_string_gets_an_ampersand() -> None:
    url = _settings("postgresql+psycopg://u:p@h/db?sslmode=require").sqlalchemy_url

    assert url == f"postgresql+psycopg://u:p@h/db?sslmode=require&{_SEARCH_PATH}"
    assert url.count("?") == 1


def test_url_with_several_existing_parameters_is_appended_to() -> None:
    url = _settings(
        "postgresql+psycopg://u:p@h/db?sslmode=require&connect_timeout=5"
    ).sqlalchemy_url

    assert url.endswith(f"&{_SEARCH_PATH}")
    assert "sslmode=require" in url
    assert "connect_timeout=5" in url
    assert url.count("?") == 1


def test_search_path_is_percent_encoded() -> None:
    """The `-c search_path=<schema>` libpq option contains a space and an
    `=`; both have to survive as encoded bytes inside the query string."""
    url = _settings("postgresql+psycopg://u:p@h/db").sqlalchemy_url

    assert "%20" in url  # the space in "-c search_path"
    assert "%3D" in url  # the "=" in "search_path=myschema"
    assert " " not in url


def test_schema_name_is_carried_into_the_search_path() -> None:
    url = Settings(
        postgres_url="postgresql+psycopg://u:p@h/db", postgres_schema="other_schema"
    ).sqlalchemy_url

    assert "search_path%3Dother_schema" in url


def test_postgres_url_is_left_otherwise_untouched() -> None:
    """Only the separator and the options parameter are added - the DSN
    itself (driver, credentials, host, database) is passed through."""
    dsn = "postgresql+psycopg://user:pw@db.internal:5433/qmd"

    assert _settings(dsn).sqlalchemy_url.startswith(f"{dsn}?")
