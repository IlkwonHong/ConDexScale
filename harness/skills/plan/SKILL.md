---
name: plan
description: Turn an agreed research idea into a falsifiable, reproducible experiment plan.
disable-model-invocation: true
---
# Plan experiment

Input: $ARGUMENTS

Read harness/memory/research_context.md and applicable decision summaries. Build a proposed experiment plan using harness/templates/plan.md, saved as harness/research/IDEA-NNN-slug/experiments/EXP-NNN-slug/plan.md. Use the next unused global EXP number (check all idea folders). If the plan reuses another idea's experiment, reference it by EXP number instead of copying it.

Include: hypothesis, baselines, equal-budget comparison, independent metrics, dataset and simulator versions, train/evaluation seeds, success criterion fixed before seeing results, ablations, anticipated failure modes, time/GPU budget, early stop conditions, and a 5-minute smoke test.

Explain how this could fail and what would be learned from a negative result.

Present plan for explicit researcher approval. Do not implement, rewrite the evaluation protocol, launch costly runs, or claim the plan is approved.
