#!/usr/bin/env python3
"""Basic static checks; does not substitute for `claude plugin validate`."""
import json
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
REPO = HARNESS.parent
required = [
    REPO / 'CLAUDE.md', HARNESS / 'CLAUDE.md',
    HARNESS / '.claude-plugin/plugin.json',
    HARNESS / 'memory/research_context.md',
    HARNESS / 'scripts/start.sh',
]
errors = [f'Missing: {p}' for p in required if not p.is_file()]
try:
    manifest = json.loads((HARNESS/'.claude-plugin/plugin.json').read_text())
    if manifest.get('name') != 'research-harness':
        errors.append('Unexpected plugin name')
except Exception as e:
    errors.append(f'Invalid manifest: {e}')
for path in (HARNESS/'skills').glob('*/SKILL.md'):
    txt = path.read_text()
    if not txt.startswith('---\n') or 'description:' not in txt:
        errors.append(f'Invalid skill frontmatter: {path}')
if '@harness/CLAUDE.md' not in (REPO/'CLAUDE.md').read_text():
    errors.append('Root CLAUDE.md missing harness import')
import re
seen = {}
for d in sorted((HARNESS/'research').glob('*/')):
    if d.name == 'references':
        continue
    if not re.fullmatch(r'IDEA-\d{3}-[a-z0-9-]+', d.name):
        errors.append(f'Idea folder name should be IDEA-NNN-short-slug: {d.name}')
    for e in sorted((d/'experiments').glob('*/')):
        m = re.fullmatch(r'(EXP-\d{3})-[a-z0-9-]+', e.name)
        if not m:
            errors.append(f'Experiment folder name should be EXP-NNN-short-slug: {e}')
            continue
        if m.group(1) in seen:
            errors.append(f'Duplicate experiment number {m.group(1)}: {seen[m.group(1)]} and {e}')
        seen[m.group(1)] = e
if errors:
    for err in errors: print('ERROR:', err)
    raise SystemExit(1)
print('PASS: research harness static validation')
