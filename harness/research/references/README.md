# References (shared by all ideas)

Papers found so far, by the researcher or by Claude, kept in one place so every discussion can search past findings, not only new searches.

```
references/
├── index.md     one entry per paper (the file to search first)
├── papers/      PDF files, named by arXiv ID or first-author-year (gitignored: large)
└── surveys/     literature survey reports and notes (markdown, tracked by git)
```

## Rules
- **One entry per paper in `index.md`**, using the entry template there. Give each paper a short key (`firstauthor-year-keyword`, e.g. `chen-2025-dexmachina`); ideas, plans and surveys cite papers by this key.
- **Verification status is required.** `verified` = Claude or the researcher opened the paper itself (PDF or official page). `unverified` = known only from search snippets, another paper's citation, or memory. Do not cite an unverified entry as fact.
- **Record who found it and when** (researcher / Claude, date), and which ideas it relates to (`IDEA-NNN`).
- **Never delete entries.** If a paper turns out irrelevant or wrongly summarized, correct the entry and note the change.
- Before proposing an idea or claiming novelty, search `index.md` and `surveys/` first, then search the web for anything newer.
