---
name: close-loop
description: Preserve research decisions and negative results after the researcher reviews the evidence.
disable-model-invocation: true
---
# Close the research loop

Input: $ARGUMENTS

After the researcher reviews results and agrees with the summary:
- Append the final decision, rationale, evidence links (idea/experiment/run folder paths), rejected alternatives, and next open question to harness/memory/decision_log.md. This is the only place decisions are written; in the idea's discussion.md add only a link to the entry.
- Update harness/memory/idea_backlog.md without deleting old ideas or negative results. If the researcher decides to pursue a candidate, create harness/research/IDEA-NNN-slug/ from templates (same IDEA number) and mark the backlog entry as promoted.
- Update the Status field in idea.md / plan.md where relevant.
- Ensure an analysis report exists for the run; update experiment status/links where relevant.
- Do not invent publications, citations, measurements or researcher approval.
- Finish with a concise handoff for the next session.
