import json
import pytest
from pathlib import Path
import subprocess
import sys
import re


def clean_json(json_str):
    """Remove comments and trailing commas from JSON string"""
    # Remove comments
    json_str = re.sub(r"//.*?\n|/\*.*?\*/", "", json_str, flags=re.S)
    # Remove trailing commas
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    return json_str


def load_launch_configurations():
    """Load configurations from launch.json"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    launch_json_path = project_root / ".vscode" / "launch.json"

    with open(launch_json_path) as f:
        json_str = f.read()
        # Clean the JSON before parsing
        clean_str = clean_json(json_str)
        data = json.loads(clean_str)

    # Get only CLI configurations
    return [
        config
        for config in data["configurations"]
        if config["type"] == "debugpy" and "cli.py" in config.get("program", "")
    ]


def run_single_configuration(config):
    """Run a single configuration and return success/failure"""
    override_params = {"num_steps": 10, "wandb_tracking": "disabled", "batch_size": 32}

    # Get CLI arguments and override parameters
    if "args" not in config:
        return f"Skipped {config['name']}: No CLI arguments found"

    args = config["args"].copy()

    # Apply overrides
    for key, value in override_params.items():
        try:
            param_index = args.index(f"--{key}")
            args[param_index + 1] = str(value)
        except ValueError:
            args.extend([f"--{key}", str(value)])

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    cli_path = project_root / "grokking" / "cli.py"

    # Run the command
    cmd = [sys.executable, str(cli_path)] + args

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(project_root)
        )

        if result.returncode != 0:
            return (
                f"Configuration {config['name']} failed:\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        return None

    except subprocess.TimeoutExpired:
        return f"Configuration {config['name']} timed out after 5 minutes"
    except subprocess.SubprocessError as e:
        return f"Configuration {config['name']} failed to run: {str(e)}"


def test_all_configurations():
    """Test all configurations from launch.json"""

    configs = load_launch_configurations()

    # Store all failures
    failures = []

    # Run each configuration
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        error = run_single_configuration(config)
        if error:
            failures.append(error)

    # If any failures occurred, fail the test with details
    if failures:
        failure_msg = "\n\n".join(failures)
        pytest.fail(f"Some configurations failed:\n\n{failure_msg}")
