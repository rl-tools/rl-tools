# Changes to latest release

- Environments should inherit from the `rl_tools::rl::environments::Environment` tag now. This allows default no-op implementations of e.g. `init` `malloc` etc. to avoid boilerplate
- `rl_tools/containers.h` has been moved to `rl_tools/containers/matrix/matrix.h`
    - A more generic structure supporting the new N-dimensional `Tensor` datastructure