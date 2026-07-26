"""Core task storage for Tasknote - a SQLite-backed task list.

Opens the database connection once per process and runs a small startup
migration that adds any columns missing from an older schema version.
"""

import sqlite3
from dataclasses import dataclass
from datetime import date


@dataclass
class Task:
    id: int
    title: str
    tag: str | None
    priority: str
    done: bool
    due_date: date | None


def get_connection(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            tag TEXT,
            priority TEXT NOT NULL DEFAULT 'medium',
            done INTEGER NOT NULL DEFAULT 0,
            due_date TEXT
        )
        """
    )
    _migrate_schema(connection)
    return connection


def _migrate_schema(connection: sqlite3.Connection) -> None:
    """Adds columns missing from an older database file, one at a time -
    there's no dedicated migration tool for a project this small."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
    if "due_date" not in columns:
        connection.execute("ALTER TABLE tasks ADD COLUMN due_date TEXT")


def add_task(
    connection: sqlite3.Connection, title: str, tag: str | None = None, priority: str = "medium"
) -> int:
    cursor = connection.execute(
        "INSERT INTO tasks (title, tag, priority) VALUES (?, ?, ?)", (title, tag, priority)
    )
    connection.commit()
    return cursor.lastrowid or 0


def complete_task(connection: sqlite3.Connection, task_id: int) -> None:
    connection.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))
    connection.commit()


def list_tasks(connection: sqlite3.Connection, tag: str | None = None) -> list[Task]:
    query = "SELECT id, title, tag, priority, done, due_date FROM tasks"
    params: tuple[str, ...] = ()
    if tag is not None:
        query += " WHERE tag = ?"
        params = (tag,)
    rows = connection.execute(query, params).fetchall()
    return [
        Task(id=r[0], title=r[1], tag=r[2], priority=r[3], done=bool(r[4]), due_date=r[5])
        for r in rows
    ]
