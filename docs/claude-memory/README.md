# Claude Code memory (backup)

These are the persistent memory files Claude Code accumulated for this project.
They normally live **outside** the repo, so they are not part of a `git clone`.
This folder is a committed backup so the context survives a machine change.

## Restore on a new machine

Claude Code looks for memory at:

    ~/.claude/projects/<project-path-slug>/memory/

where `<project-path-slug>` is the project's absolute path with `/` replaced by `-`.
For example, if you clone to `/Users/you/Downloads/crypto-trading-agent`, the slug is
`-Users-you-Downloads-crypto-trading-agent`.

To restore:

    SLUG=$(pwd | sed 's#/#-#g')
    mkdir -p "$HOME/.claude/projects/$SLUG/memory"
    cp docs/claude-memory/*.md "$HOME/.claude/projects/$SLUG/memory/"

`MEMORY.md` is the index loaded each session; the `project_*.md` / `user_*.md`
files are the individual memories it points to.
