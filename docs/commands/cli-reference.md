# CLI reference

Generated directly from marq's actual `click` command tree
(`qmd_py.cli.main:cli`) — this page can't drift out of sync with the
code the way a hand-transcribed `--help` dump would.

!!! note
    `deep-search` is a registered alias for `query` (same command
    object, two names). It appears below under a second "marq query"
    heading rather than "marq deep-search" — the generator names each
    section after the command object's own name, not the alias it was
    registered under. Both names work identically on the command line.

::: mkdocs-click
    :module: qmd_py.cli.main
    :command: cli
    :prog_name: marq
    :depth: 1
    :style: table
