# Git Commit Workflow

Professional git workflow for Kaggle competition projects.

## Pre-commit Checks

First, review current changes:

```bash
!git status
!git diff --staged
```

## TODO.md Verification

**CRITICAL**: Before committing, ensure @todo.md has been updated:

- [ ] Current task is documented
- [ ] Work is marked as completed
- [ ] Next steps are identified

## Branch Naming Conventions

Choose appropriate branch name:

- `feature/description` - New functionality
- `experiment/exp-name` - ML experiment
- `fix/bug-description` - Bug fix
- `docs/update-description` - Documentation update
- `chore/task-description` - Maintenance, dependencies

## Conventional Commits Format

Use structured commit messages:

### Format

```
type: Short description (imperative mood)

Optional detailed explanation of changes.

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `exp`: Experiment results (with metrics)
- `docs`: Documentation changes
- `chore`: Dependencies, config, tooling
- `refactor`: Code restructuring
- `test`: Test additions/modifications
- `perf`: Performance improvements

### Examples

```bash
feat: Add ConvNeXt baseline model with timm

exp: Record 5-fold CV results (R²=0.58, RMSE=0.23)

fix: Resolve CUDA OOM by reducing batch_size to 16

docs: Update TODO with Phase 2 experiment plan

chore: Update PyTorch to 2.1.2 for Kaggle compatibility
```

## Stage and Commit

```bash
# Stage specific files (use $ARGUMENTS)
git add $ARGUMENTS

# OR stage all changes
git add .

# Create commit with message
git commit -m "$(cat <<'EOF'
type: Description

Detailed explanation if needed.

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

## Post-commit Verification

```bash
!git status
!git log -1 --stat
```

## Pre-commit Hook Failures

If pre-commit hooks modify files:

1. Review changes: `git diff`
2. Verify safety: `git log -1 --format='[%h] (%an <%ae>) %s'`
3. If it's your commit: `git commit --amend --no-edit`
4. If not: Create new commit

## Push to Remote

```bash
# First push (create tracking branch)
git push -u origin $(git branch --show-current)

# Subsequent pushes
git push
```

## Important Reminders

- ✅ Always update @todo.md before committing
- ✅ Use descriptive commit messages (WHY, not WHAT)
- ✅ Keep commits focused and atomic
- ❌ Never use --no-verify unless absolutely necessary
- ❌ Never force push to main/master
- ❌ Never commit secrets or large data files
