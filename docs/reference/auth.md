# `qmd_py.auth`

The ACL mock: `get_current_user()` and the `can_access()` choke point —
see [Architecture › ACL](../architecture.md#acl) for why this exists as
real, structured plumbing even though the check itself is mocked True
today.

::: qmd_py.auth
    options:
      filters: ["!^__"]
