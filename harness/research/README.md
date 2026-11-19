# Research folder

Each approved idea gets one folder. Experiments live inside the idea that created them; runs live inside their experiment.

```
research/
├── references/            papers shared by all ideas (see references/README.md)
└── IDEA-NNN-short-slug/
    ├── idea.md            (templates/idea.md)
    ├── discussion.md      (templates/discussion.md; reasoning only)
    └── experiments/
        └── EXP-NNN-short-slug/
            ├── plan.md        (templates/plan.md)
            ├── analysis.md    (templates/analysis.md)
            └── runs/run-NNN/  (written by scripts/run_experiment.py)
```

## Rules
- **Candidates stay in `harness/memory/idea_backlog.md`.** Create an `IDEA-NNN-*` folder only when the researcher decides to pursue the idea. Keep the same IDEA number as the backlog entry.
- **Names:** number plus a short lowercase slug, e.g. `IDEA-002-dynamic-retarget`, `EXP-004-arctic-reprojection-check`.
- **EXP numbers are global across all ideas and never reused.** An experiment shared by several ideas stays in the folder of the idea that created it; other ideas reference it by EXP number in `idea.md` and `plan.md`.
- **Decisions are written only in `harness/memory/decision_log.md`**, with a link to the idea or experiment folder. `discussion.md` keeps the reasoning and links to the decision entry.
- **Runs:** git tracks only `run.json` (command, git commit, dirty state, time, exit code) and `output.log`. Checkpoints, videos and other large files in a run folder are gitignored; record their location in `analysis.md`.
- Never delete or rewrite old runs, negative results or dropped ideas; change their status instead.
