# Changes to latest release

- Environments should inherit from the `rl_tools::rl::environments::Environment` tag now. This allows default no-op implementations of e.g. `init` `malloc` etc. to avoid boilerplate