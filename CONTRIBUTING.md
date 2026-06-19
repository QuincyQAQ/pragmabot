# Contributing to PragmaBot

We welcome contributions! Here's how to get started.

## Reporting Issues

- Search [existing issues](https://github.com/leggedrobotics/pragmabot/issues) before opening a new one.
- Include your OS, Python version, ROS version, and steps to reproduce.

## Development Setup

1. Fork the repository and clone your fork.
2. Install dependencies: `pip install -r requirements.txt`
3. Build the ROS package following the [README](README.md#installation).

## Pull Requests

1. Open an issue first to discuss your proposed change.
2. Create a feature branch from `main`.
3. Keep commits focused — one logical change per commit.
4. Ensure the code follows the existing style (PEP 8, type hints, Google-style docstrings).
5. Test your changes locally before submitting.
6. Open a pull request with a clear description of the change and its motivation.

## Code Style

- Python: PEP 8, type hints on all public functions, Google-style docstrings.
- ROS: Follow [ROS best practices](https://wiki.ros.org/BestPractices) for naming and structure.
- Use `logging` instead of `print()` in library/node code.

## License

By contributing, you agree that your contributions will be licensed under the [BSD 3-Clause License](LICENSE).
