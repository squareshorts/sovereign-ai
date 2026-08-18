
import os
import subprocess
import sys
import shutil

def test_full_analyzer_mock():
    # If there are no results, run them
    env = os.environ.copy()
    env['OPENAI_API_KEY'] = 'dummy'
    env['ANTHROPIC_API_KEY'] = 'dummy'
    env['GEMINI_API_KEY'] = 'dummy'

    # Run the experiment
    if os.path.exists('results/provider_switch'):
        shutil.rmtree('results/provider_switch')
    cmd = [sys.executable, 'experiments/provider_switch/run_experiment.py', '--mock-adapters', '--formal']
    subprocess.check_call(cmd, env=env)

    # Run the analyzer with patched bootstrap count for tests
    import tempfile
    with open('experiments/provider_switch/analyze_results.py', 'r') as f:
        content = f.read()
    content = content.replace('PRODUCTION_BOOTSTRAP_RESAMPLES = 10000', 'PRODUCTION_BOOTSTRAP_RESAMPLES = 10')
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as f:
        f.write(content)
        temp_analyzer = f.name
        
    cmd_analyzer = [sys.executable, temp_analyzer]
    try:
        subprocess.check_call(cmd_analyzer)
    finally:
        os.remove(temp_analyzer)

    # Check that figures and csvs are generated
    assert os.path.exists('results/provider_switch/replicate_summary.csv')
    assert os.path.exists('results/provider_switch/migration_ci.csv')
    assert os.path.exists('results/provider_switch/performance_ci_plot.png')
    assert os.path.exists('results/provider_switch/migration_sequence_plot.png')
    assert os.path.exists('results/provider_switch/provider_agreement_plot.png')
