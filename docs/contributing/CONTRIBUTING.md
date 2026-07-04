# Contributing Guide

Thank you for contributing to **The Searchlight Protocol**! This document outlines guidelines for code quality, formatting, pull requests, and setup targets.

---

## 🛠️ Style Guide and Code Guidelines

To keep the codebase uniform and readable, please adhere to the following standards when writing Python or React code:

### 1. Python Style Guidelines
*   **Code Formatting**: We use PEP-8 standards. Run `black` or `ruff` to auto-format code files before submitting a pull request.
*   **Imports Order**: Group imports into three blocks separated by blank lines:
    1.  Standard library modules.
    2.  Third-party libraries (PyTorch, OpenCV, FastAPI, etc.).
    3.  Local module imports.
*   **Documentation style**: Write Google-style Python docstrings for all new functions, methods, modules, and classes.

### 2. React Style Guidelines
*   **JSX Components**: Wrap all core components in `memo()` where appropriate to optimize render performance.
*   **Prop Documentation**: Write clear JSDoc annotations above all component declarations detailing parameter types.
*   **Tailwind Style Guidelines**: Reuse global CSS tokens inside `index.css` rather than nesting inline arbitrary styles.

---

## ⚙️ Development Workflows

### 1. Branch Naming Conventions
Create descriptive branches based on the work type:
*   `feature/your-feature-name`
*   `bugfix/issue-description`
*   `docs/documentation-update`

### 2. Pull Request Checklist
Before opening a pull request, verify:
*   [ ] The codebase runs locally (both frontend and backend).
*   [ ] Unused imports have been pruned.
*   [ ] Python files pass syntax check (`python -m py_compile ...`).
*   [ ] All new parameters are updated inside the environment configuration files.
*   [ ] The changes do not introduce regressions to model warmup steps.
