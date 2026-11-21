# Research Harness (Claude Code plugin)

Copy the **contents of this starter directory** into an existing research repository. Only `CLAUDE.md` is at repository root; all harness-specific components are under `harness/`.

## Quick start

From the repository root:

```bash
bash harness/scripts/start.sh discuss
# Then inside Claude Code: /research-harness:discuss
```

Use `discuss` for read-only deliberation. When ready to implement, start a normal permission session (or consciously change permission mode in your existing session) and run:

```bash
bash harness/scripts/start.sh execute
# Inside Claude Code: /research-harness:plan
# After explicit approval: /research-harness:implement
# When runs finish: /research-harness:analyze
# Then: /research-harness:close-loop
```

Skills live at `harness/skills/`. This plugin is explicitly loaded through `--plugin-dir` in `start.sh`. It does **not** install into the account permanently. If skills change mid-session, use `/reload-plugins` in compatible Claude Code versions.

Plan mode is read-only, so to persist a discussion's decision log, first approve the resulting research plan and switch into a normal permission session. The researcher's explicit approval is required; a skill prompt by itself does not provide OS-level enforcement.

## Record and inspect experiment execution

```bash
python harness/scripts/run_experiment.py --exp EXP-001 -- python -c "print('smoke test')"
python harness/scripts/check_harness.py
```

The experiment folder (`harness/research/IDEA-NNN-slug/experiments/EXP-001-slug/` with `plan.md`) must exist first. The wrapper saves `run.json` and `output.log` to its `runs/run-NNN/`. See `harness/research/README.md` for the folder rules. It does not run anything in the background, schedule GPU tasks, or approve experiments. Use it for commands you have already decided to run; adapt to Slurm/Kubernetes if needed. Inside run folders only `run.json` and `output.log` are tracked; everything else is gitignored by `harness/.gitignore`.

## Recommended project adaptation

1. Fill in `harness/memory/research_context.md` with actual problem, dataset, hardware, baseline, and benchmark.
2. Replace the sample train/eval placeholders in `harness/templates/plan.md`.
3. Decide your project's GPU/Slurm approval and max budget policy outside the LLM.
4. Commit research hypotheses and key decisions, but not raw datasets, weights, or large rollout videos.
5. Use Git worktrees or branches for risky algorithm changes.

## Verification

- `python harness/scripts/check_harness.py` checks the directory structure, JSON and skill frontmatter.
- `claude plugin validate ./harness` checks the actual plugin when your installed Claude Code supports it.
- `/context all` verifies root CLAUDE.md import in Claude Code; `/help` or skill completion checks plugin commands.

References: https://code.claude.com/docs/en/plugins/create and https://code.claude.com/docs/en/memory
