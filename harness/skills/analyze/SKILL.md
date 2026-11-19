---
name: analyze
description: Audit experiment evidence, report uncertainty, failure modes, and next experiments.
disable-model-invocation: true
---
# Analyze experimental results

Input: $ARGUMENTS

Read the corresponding plan, raw logs and run metadata, not merely summary text. Check matching datasets, seeds, simulator versions, evaluation rules and budgets.

Report observed metrics, within-seed paired differences when available, variation/intervals, baselines, sample size, suspected confounders, failure patterns and negative results. Keep claims proportional to evidence; separate observed improvements from plausible explanations.

Propose up to three targeted next experiments with resource estimates and information gained. Never change evaluation rules after looking at results without clearly identifying an exploratory follow-up.
Use harness/templates/analysis.md for the final report, saved as analysis.md in the experiment folder (next to plan.md and runs/).
