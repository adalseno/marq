# Architecture

Tasknote is deliberately tiny: a Python core, a thin JavaScript HTTP API
in front of it, and TypeScript types shared with the (imaginary) frontend.

## Storage

All tasks live in a single SQLite file. `src/tasks.py` opens the database
connection on startup and runs a one-shot migration that adds any columns
missing from an older schema version - there's no separate migration
tool, just a version check against a `schema_version` table.

## API layer

`src/api.js` is a thin HTTP wrapper: each route parses the request,
calls into the Python core over a local socket, and serializes the
response as JSON. It does no business logic of its own.

## Types

`src/models.ts` mirrors the shapes returned by the API so the frontend
gets type-checked access to `Task`, `Tag`, and `Priority` without
duplicating the definitions by hand.
