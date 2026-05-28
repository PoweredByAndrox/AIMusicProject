# Git AI Commit Rules

All commits created by an AI agent must follow Conventional Commits.

## Format

```txt
<type>(<scope>): <summary>
```

Examples:

```txt
feat(auth): add login form validation
fix(download): prevent duplicate file verification
refactor(ui): split launcher layout into smaller components
docs(readme): update local setup instructions
test(verifier): add checksum mismatch coverage
chore(config): update electron build settings
```

## Allowed Types

Use only:

- `feat`: new user-facing feature
- `fix`: bug fix
- `refactor`: internal code change without behavior change
- `perf`: performance improvement
- `docs`: documentation-only change
- `test`: tests only
- `style`: formatting, whitespace, lint fixes without logic changes
- `chore`: maintenance work
- `build`: dependencies, bundlers, build scripts, package config
- `ci`: CI configuration only
- `revert`: revert a previous commit

## Scope Rules

The scope must describe the affected module, folder, feature, or domain.

Good scopes:

```txt
auth, ui, api, launcher, downloader, verifier, config, docs, tests
```

Bad scopes:

```txt
stuff, misc, changes, update, fix
```

## Summary Rules

The summary must:

- Be written in English.
- Be concise and specific.
- Use imperative mood.
- Be lowercase except for proper names.
- Be no longer than 72 characters.
- Not end with a period.
- Describe the actual change, not the intention.
- Not mention AI, ChatGPT, Codex, Cursor, prompts, or conversations.

## Required Workflow Before Committing

Before creating any commit, the agent must:

1. Run and inspect `git status --short`.
2. Run and inspect `git diff`.
3. Run and inspect `git diff --staged` if anything is staged.
4. Decide whether the changes belong to one logical commit.
5. Split unrelated changes into separate commits or ask before committing.
6. Never commit files that were not inspected.

## Staging Rules

Do not blindly run:

```bash
git add .
```

Prefer explicit paths:

```bash
git add path/to/file
```

or interactive staging:

```bash
git add -p
```

## Never Commit

- secrets
- API keys
- `.env` files
- debug logs
- temporary files
- local IDE/cache files
- generated junk
- unrelated changes
- broken experiments
