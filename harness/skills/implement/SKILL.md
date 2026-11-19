---
name: implement
description: Implement a researcher's approved experiment with tests and minimal code changes.
disable-model-invocation: true
---
# Implement an approved experiment

Input: $ARGUMENTS

1. Confirm the human researcher approved a specific experiment plan. If not, stop and request approval.
2. Read harness/memory/research_context.md and the approved plan.
3. Inspect current git status; do not discard unrelated user modifications.
4. Implement the smallest isolated change necessary, keeping baseline and evaluation protocol unchanged.
5. Add/modify tests and run a short smoke test. State test commands and results.
6. Before expensive training, show exact launch command, job counts, resource and cost estimates and request approval separately.
7. For approved runs, use the experiment wrapper where feasible; retain configs, commit ID, logs, seed and version information.
8. Report files modified, tests passed/failed, and exact next steps. Do not imply tests passed if not run.
