# Tasknote

A small task-tracking app used as this project's frozen test fixture -
just a handful of files, not a real product. Tasks have a title, an
optional tag, a priority (low/medium/high), and a due date.

## Features

- Add, complete, and list tasks from the CLI or the HTTP API
- Tag-based filtering (e.g. "work", "personal")
- SQLite-backed storage with a simple migration on startup

## Layout

- `src/tasks.py` - core task storage and the database connection
- `src/api.js` - a small HTTP API exposing the task list
- `src/models.ts` - shared TypeScript types for the frontend
- `docs/architecture.md` - how the pieces fit together
- `CHANGELOG.md` - release notes
