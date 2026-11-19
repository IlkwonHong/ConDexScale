# Experiments

Create one plan (e.g., EXP-001.md) and one analysis report (e.g., EXP-001-analysis.md) per experiment. The runner stores immutable-ish execution logs in `runs/`, which are ignored by git. Keep concise machine-readable metadata and links to results in your existing experiment tracker or add an experiment index here.

The log wrapper intentionally does not submit jobs to Slurm or allocate hardware. It records the command supplied by the user/agent, git revision, working tree state, runtime timestamps and exit status. It is not a security guard or approval system.
