
import os
import subprocess
import sys
import shutil

def test_resume_mock():
    # clean up previous run
    if os.path.exists('results/provider_switch'):
        shutil.rmtree('results/provider_switch')

    env = os.environ.copy()
    env['OPENAI_API_KEY'] = 'dummy'
    env['ANTHROPIC_API_KEY'] = 'dummy'
    env['GEMINI_API_KEY'] = 'dummy'
    env['ABORT_AFTER_UNITS'] = '347'

    cmd = [sys.executable, 'experiments/provider_switch/run_experiment.py', '--mock-adapters']
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError:
        pass  # expected to abort

    # Verify 347 lines
    with open('results/provider_switch/raw_outputs.jsonl', 'r') as f:
        lines = f.readlines()
    assert len(lines) == 347

    # Run resume
    env.pop('ABORT_AFTER_UNITS', None)
    cmd_resume = [sys.executable, 'experiments/provider_switch/run_experiment.py', '--mock-adapters', '--formal', '--resume']
    subprocess.check_call(cmd_resume, env=env)

    with open('results/provider_switch/raw_outputs.jsonl', 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 2160
