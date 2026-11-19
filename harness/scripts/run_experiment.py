#!/usr/bin/env python3
"""Record a foreground training/evaluation command with basic provenance.

Usage: python harness/scripts/run_experiment.py --exp EXP-001 -- python train.py --seed 0

Writes harness/research/IDEA-*/experiments/EXP-001-*/runs/run-NNN/{run.json,output.log}.

The command is run synchronously and inherits the current environment. Do not pass
secrets in command-line arguments. This is a logger, NOT an authorization boundary.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

HARNESS = Path(__file__).resolve().parents[1]
PROJECT = HARNESS.parent

def git(args):
    result = subprocess.run(['git', *args], cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.stdout.strip() if result.returncode == 0 else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True,
                        help='Experiment number, e.g. EXP-003 (its folder must already exist under research/*/experiments/)')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required after --')
    m = re.fullmatch(r'(EXP-\d{3})(-[a-z0-9-]+)?', args.exp)
    if not m:
        parser.error('--exp must look like EXP-003 or EXP-003-short-slug')
    matches = sorted(p for p in (HARNESS / 'research').glob(f'IDEA-*/experiments/{m.group(1)}*')
                     if p.is_dir() and re.fullmatch(re.escape(m.group(1)) + r'(-.*)?', p.name))
    if len(matches) != 1:
        parser.error(f'Expected exactly one folder for {m.group(1)}, found {len(matches)}: '
                     f'{[str(p.relative_to(HARNESS)) for p in matches]}. Create the experiment folder with plan.md first.')
    exp_dir = matches[0]
    runs = exp_dir / 'runs'
    runs.mkdir(exist_ok=True)
    while True:  # next sequential run-NNN; never overwrite an existing run
        nums = [int(p.name[4:]) for p in runs.glob('run-*') if p.name[4:].isdigit()]
        run_dir = runs / f'run-{max(nums, default=0) + 1:03d}'
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    meta = {
        'experiment_id': exp_dir.name,
        'idea_id': exp_dir.parents[1].name,
        'run_id': run_dir.name,
        'run_uid': uuid.uuid4().hex[:6],
        'command': command,
        'project_directory': str(PROJECT),
        'utc_started': datetime.now(timezone.utc).isoformat(),
        'git_commit': git(['rev-parse', 'HEAD']),
        'git_dirty': bool(git(['status', '--porcelain'])),
        'python_version': sys.version.split()[0],
        'hostname': os.uname().nodename if hasattr(os, 'uname') else None,
        'exit_code': None,
    }
    print(f'Run directory: {run_dir}', flush=True)
    (run_dir/'run.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    rc = 1
    try:
        with (run_dir/'output.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = process.wait()
    except OSError as exc:
        (run_dir/'output.log').write_text(f'Failed to launch: {exc}\n')
        print(f'Failed to launch: {exc}', file=sys.stderr)
        rc = 127
    finally:
        meta['utc_finished'] = datetime.now(timezone.utc).isoformat()
        meta['exit_code'] = rc
        (run_dir/'run.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return rc

if __name__ == '__main__':
    sys.exit(main())
