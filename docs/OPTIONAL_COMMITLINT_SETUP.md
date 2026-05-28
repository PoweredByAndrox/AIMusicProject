# Optional Commitlint Setup

The AI rules tell Codex and Cursor how to write commits, but they do not technically block invalid commit messages.

For hard local validation, install Commitlint and Husky.

## Install

```bash
npm install --save-dev @commitlint/cli @commitlint/config-conventional husky
```

## Create Commitlint Config

Create `commitlint.config.cjs` in the project root:

```js
module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    "type-enum": [
      2,
      "always",
      [
        "feat",
        "fix",
        "refactor",
        "perf",
        "docs",
        "test",
        "style",
        "chore",
        "build",
        "ci",
        "revert",
      ],
    ],
    "subject-case": [0],
    "header-max-length": [2, "always", 72],
  },
};
```

## Enable Husky

```bash
npx husky init
```

Create or edit `.husky/commit-msg`:

```bash
npx --no -- commitlint --edit "$1"
```

Invalid commit messages will then be rejected locally.
