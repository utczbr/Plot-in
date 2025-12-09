# Contributing to the Chart Analysis System

We welcome contributions to improve the Chart Analysis System. To ensure a smooth and effective development process, please adhere to the following guidelines.

## 1. Coding Standards

This project follows a strict set of coding standards to maintain code quality, readability, and maintainability.

- **Style Guide**: All Python code must be compliant with [PEP 8](https://www.python.org/dev/peps/pep-0008/). We use automated tools to enforce this.
- **Formatting**: We use `black` for code formatting and `isort` for import sorting. Please run these tools on your code before submitting a contribution.
- **Type Hinting**: All function and method signatures must include type hints for all arguments and return values, as specified in [PEP 484](https://www.python.org/dev/peps/pep-0484/).
- **Docstrings**: Every module, class, and public function/method must have a comprehensive docstring that follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#3.8-comments-and-docstrings). Docstrings should include:
    - A brief one-line summary.
    - A more detailed description of the object's purpose.
    - `Args:` section for all arguments.
    - `Returns:` section describing the return value.
    - `Raises:` section for any exceptions that can be raised.
- **Naming Conventions**: Use `snake_case` for functions, methods, and variables. Use `PascalCase` for classes.
- **Code Quality**:
    - **No Temporary Flags**: Do not commit any temporary flags, such as `DEBUG_MODE` or `TEMP_FIX`. Use the `logging` module with appropriate levels (`DEBUG`, `INFO`, `WARNING`) for debugging information.
    - **No Commented-Out Code**: Remove all commented-out code blocks before committing. Code that is not used should be deleted.
    - **Magic Numbers**: Avoid magic numbers. Use named constants or configuration parameters instead.

## 2. Pull Request (PR) Process

1.  **Fork the Repository**: Start by forking the main repository.
2.  **Create a Branch**: Create a new branch from `main` for your feature or bug fix. Use a descriptive name (e.g., `feature/new-chart-handler` or `fix/orchestrator-bug`).
3.  **Develop**: Make your changes, adhering to the coding standards above.
4.  **Test**: Add or update unit/integration tests to cover your changes. Ensure that all existing tests pass.
5.  **Lint and Format**: Run `black` and `isort` on your code to ensure it is formatted correctly.
    ```bash
    black .
    isort .
    ```
6.  **Submit a Pull Request**: Push your branch to your fork and open a pull request to the `main` branch of the original repository.
7.  **Code Review**: Your PR will be reviewed by maintainers. Be prepared to address feedback and make changes.

## 3. Testing Requirements

- **Unit Tests**: All new functions and classes should have corresponding unit tests.
- **Integration Tests**: For larger features that involve multiple components, add integration tests to ensure the components work together correctly.
- **Test Coverage**: Aim to maintain or increase the existing test coverage.
- **Running Tests**: A script or command for running the test suite should be provided (e.g., `pytest`). Ensure all tests pass before submitting a PR.
  ```bash
  pytest
  ```
  *(Note: The testing framework and commands need to be formally defined for the project.)*
