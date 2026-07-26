# Changelog

## 0.3.0

- Added `due_date` to tasks, with an on-startup schema migration for
  existing databases.
- `GET /tasks?tag=` now filters by tag.

## 0.2.0

- Added priority levels (low/medium/high) to tasks.

## 0.1.0

- Initial release: add/complete/list tasks, SQLite storage.
