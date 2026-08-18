
import os
import subprocess
import sys
import shutil

def test_full_production_mock():
    # clean up previous run
    if os.path.exists('results/provider_switch'):
        shutil.rmtree('results/provider_switch')

    env = os.environ.copy()
    env['OPENAI_API_KEY'] = 'dummy'
    env['ANTHROPIC_API_KEY'] = 'dummy'
    env['GEMINI_API_KEY'] = 'dummy'

    cmd = [sys.executable, 'experiments/provider_switch/run_experiment.py', '--mock-adapters', '--formal']
    subprocess.check_call(cmd, env=env)

    # Verify 2160 lines in raw_outputs.jsonl
    with open('results/provider_switch/raw_outputs.jsonl', 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 2160
