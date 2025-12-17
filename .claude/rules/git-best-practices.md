---
paths:
  - .git/**
  - "*.md"
---

# Git Best Practices (Universal)

## Branch Naming

- `feature/description` - New functionality
- `fix/bug-description` - Bug fixes
- `docs/update-description` - Documentation
- `refactor/description` - Code restructuring
- `chore/description` - Maintenance, dependencies

## Conventional Commits

Format: `type: Short description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code restructuring
- `test`: Test additions/modifications
- `chore`: Dependencies, config, tooling
- `perf`: Performance improvements

## Commit Message Template

```
type: Short description (imperative mood)

Optional detailed explanation.

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## .gitignore Essentials

```gitignore
# Python
__pycache__/
*.py[cod]
.Python
*.egg-info/
.venv/

# IDE
.vscode/
.idea/

# Environment
.env
.env.local

# Testing
.pytest_cache/
.coverage
htmlcov/

# Build
dist/
build/
```
