#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyyaml>=6.0",
#     "rich>=13.0",
#     "nibabel>=5.0",
# ]
# ///
"""
Neurocontainer Test Runner

Runs YAML-based tests for neuroimaging containers with parallel execution support.

Usage:
    ./run_tests.py                          # Run all tests
    ./run_tests.py niimath.yaml             # Run specific test file
    ./run_tests.py *.yaml -j 4              # Run with 4 parallel workers
    ./run_tests.py -l                       # List available test files
    ./run_tests.py niimath.yaml -f "smooth" # Filter tests by name pattern

Test files are loaded from tests/ directory. Tests run in work/ directory.
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Manager, Queue
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


@dataclass
class TestResult:
    """Result of a single test execution."""
    name: str
    passed: bool
    duration: float
    start_time: str = ""
    message: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class TestSuiteResult:
    """Result of a test suite (YAML file) execution."""
    name: str
    container: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration: float = 0.0
    results: list[TestResult] = field(default_factory=list)


@dataclass
class PreparedTest:
    """A test prepared for execution with all context."""
    suite_name: str
    container_name: str
    container_path: Path
    test: dict
    variables: dict[str, str]
    work_dir: Path
    global_env_setup: str | None
    default_timeout: int


def find_container(container_pattern: str, containers_dir: Path) -> Path | None:
    """Find container file matching pattern."""
    if containers_dir.exists():
        # Try exact match first
        exact = containers_dir / container_pattern
        if exact.exists():
            return exact

        # Try glob pattern
        base_name = container_pattern.replace(".simg", "").split("_")[0]
        matches = list(containers_dir.glob(f"{base_name}_*.simg"))
        if matches:
            return sorted(matches)[-1]  # Return newest version

    return None


def substitute_variables(text: str, variables: dict[str, str]) -> str:
    """Substitute ${var} placeholders with values."""
    if not text:
        return text

    result = text
    for key, value in variables.items():
        result = result.replace(f"${{{key}}}", str(value))
        result = result.replace(f"${key}", str(value))

    return result


def check_file_exists(path: str) -> bool:
    """Check if file exists."""
    return Path(path).exists()


def check_same_dimensions(path1: str, path2: str) -> tuple[bool, str]:
    """Check if two NIfTI files have same dimensions."""
    try:
        import nibabel as nib

        img1 = nib.load(path1)
        img2 = nib.load(path2)

        shape1 = img1.shape
        shape2 = img2.shape

        if shape1 == shape2:
            return True, f"Dimensions match: {shape1}"
        else:
            return False, f"Dimension mismatch: {shape1} vs {shape2}"
    except Exception as e:
        return False, f"Error comparing dimensions: {e}"


def run_single_test(
    test: dict,
    container_path: Path,
    variables: dict[str, str],
    work_dir: Path,
    global_env_setup: str | None = None,
    default_timeout: int = 120,
) -> TestResult:
    """Run a single test and return result."""
    from datetime import datetime

    name = test.get("name", "Unnamed test")
    start_timestamp = datetime.now().isoformat()
    start_time = time.time()

    try:
        # Get command and substitute variables
        command = test.get("command", "")
        if not command:
            return TestResult(
                name=name,
                passed=False,
                duration=0,
                start_time=start_timestamp,
                message="No command specified",
            )

        command = substitute_variables(command, variables)

        # Build environment setup
        env_setup = test.get("env_setup", global_env_setup) or ""
        if env_setup:
            env_setup = substitute_variables(env_setup, variables)

        # Write command to temporary script file (avoids all shell quoting issues)
        script_path = work_dir / f".test_{os.getpid()}_{int(time.time()*1e6)}.sh"
        try:
            with open(script_path, 'w') as f:
                f.write("#!/usr/bin/env bash\n")
                if env_setup:
                    f.write(f"{env_setup}\n")
                f.write(f"{command}\n")
            os.chmod(script_path, 0o755)
        except OSError as e:
            return TestResult(
                name=name, passed=False, duration=time.time() - start_time,
                start_time=start_timestamp, message=f"Failed to create test script: {e}",
            )

        try:
            if container_path:
                binds = set()
                binds.add(f"{work_dir}:{work_dir}")
                for key, value in variables.items():
                    if key not in ["output_dir"] and "/" in str(value):
                        parent = Path(value).parent
                        if parent.exists():
                            binds.add(f"{parent}:{parent}")

                cmd_list = ["apptainer", "exec", "--writable-tmpfs"]
                for b in binds:
                    cmd_list.extend(["-B", b])
                cmd_list.extend([str(container_path), "bash", str(script_path)])
            else:
                cmd_list = ["bash", str(script_path)]

            timeout = test.get("timeout", default_timeout)

            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
            )

            duration = time.time() - start_time
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        finally:
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                pass

        # Check expected exit code (default: expect success)
        expected_exit_code = test.get("expected_exit_code", 0)
        expected_exit_code_not = test.get("expected_exit_code_not")

        if expected_exit_code_not is not None:
            # expected_exit_code_not takes precedence when explicitly set
            if exit_code == expected_exit_code_not:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    start_time=start_timestamp,
                    message=f"Exit code should not be {expected_exit_code_not}",
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                )
        elif exit_code != expected_exit_code:
            return TestResult(
                name=name,
                passed=False,
                duration=duration,
                start_time=start_timestamp,
                message=f"Expected exit code {expected_exit_code}, got {exit_code}",
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )

        # Check expected output
        expected_output = test.get("expected_output_contains")
        if expected_output:
            combined_output = stdout + stderr

            if isinstance(expected_output, str):
                expected_list = [expected_output]
            else:
                expected_list = expected_output

            for expected in expected_list:
                if expected and expected not in combined_output:
                    return TestResult(
                        name=name,
                        passed=False,
                        duration=duration,
                        start_time=start_timestamp,
                        message=f"Expected output not found: '{expected[:50]}...'",
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=exit_code,
                    )

        # Run validations
        validations = test.get("validate", [])
        for validation in validations:
            if isinstance(validation, dict):
                for val_type, val_arg in validation.items():
                    if val_type == "output_exists":
                        path = substitute_variables(str(val_arg), variables)
                        if not check_file_exists(path):
                            return TestResult(
                                name=name,
                                passed=False,
                                duration=duration,
                                start_time=start_timestamp,
                                message=f"Output file not found: {path}",
                                stdout=stdout,
                                stderr=stderr,
                                exit_code=exit_code,
                            )

                    elif val_type == "same_dimensions":
                        if isinstance(val_arg, list) and len(val_arg) == 2:
                            path1 = substitute_variables(str(val_arg[0]), variables)
                            path2 = substitute_variables(str(val_arg[1]), variables)
                            ok, msg = check_same_dimensions(path1, path2)
                            if not ok:
                                return TestResult(
                                    name=name,
                                    passed=False,
                                    duration=duration,
                                    start_time=start_timestamp,
                                    message=msg,
                                    stdout=stdout,
                                    stderr=stderr,
                                    exit_code=exit_code,
                                )

        return TestResult(
            name=name,
            passed=True,
            duration=duration,
            start_time=start_timestamp,
            message="OK",
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            name=name,
            passed=False,
            duration=time.time() - start_time,
            start_time=start_timestamp,
            message=f"Timeout after {test.get('timeout', default_timeout)}s",
        )
    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            duration=time.time() - start_time,
            start_time=start_timestamp,
            message=f"Error: {e}",
        )


def _run_container_health_check(
    container_path: Path, work_dir: Path, variables: dict[str, str]
) -> TestResult:
    """Quick check that the container can execute a basic command."""
    from datetime import datetime

    start = time.time()

    binds = set()
    binds.add(f"{work_dir}:{work_dir}")

    cmd_list = ["apptainer", "exec", "--writable-tmpfs"]
    for b in binds:
        cmd_list.extend(["-B", b])
    cmd_list.extend([str(container_path), "true"])

    try:
        result = subprocess.run(
            cmd_list, capture_output=True, text=True, timeout=30, cwd=work_dir
        )
        if result.returncode == 0:
            return TestResult(
                name="Container health check",
                passed=True,
                duration=time.time() - start,
                start_time=datetime.now().isoformat(),
                message="OK",
                exit_code=0,
            )
        else:
            return TestResult(
                name="Container health check",
                passed=False,
                duration=time.time() - start,
                start_time=datetime.now().isoformat(),
                message=f"Container cannot execute commands (exit {result.returncode}): {result.stderr[:500]}",
                exit_code=result.returncode,
                stderr=result.stderr,
            )
    except Exception as e:
        return TestResult(
            name="Container health check",
            passed=False,
            duration=time.time() - start,
            start_time=datetime.now().isoformat(),
            message=f"Container health check error: {e}",
        )


def run_test_suite(
    yaml_path: Path,
    containers_dir: Path,
    work_dir: Path,
    test_filter: str | None = None,
    verbose: bool = False,
    on_test_complete: Any = None,
    result_queue: Any = None,
    running_tests: Any = None,
) -> TestSuiteResult:
    """Run all tests in a YAML file."""
    start_time = time.time()

    # Load YAML
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    suite_name = config.get("name", yaml_path.stem)
    container_name = config.get("container", "")
    default_timeout = config.get("default_timeout", 120)  # Default 2 minutes

    # Find container
    container_path = find_container(container_name, containers_dir)
    if not container_path:
        return TestSuiteResult(
            name=suite_name,
            container=container_name,
            total=0,
            failed=1,
            results=[TestResult(
                name="Container lookup",
                passed=False,
                duration=0,
                message=f"Container not found: {container_name}",
            )],
        )

    # Build variables dict
    variables = {}
    test_data = config.get("test_data", {})
    for key, value in test_data.items():
        if key == "output_dir":
            # Make output dir absolute under work_dir
            variables[key] = str(work_dir / value)
        else:
            # Make paths absolute
            path = Path(value)
            if not path.is_absolute():
                path = work_dir / value
            variables[key] = str(path)

    # Get global env setup
    global_env_setup = config.get("env_setup")

    # Clean output directory before running suite
    if "output_dir" in variables:
        output_dir = Path(variables["output_dir"])
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run setup script
    setup = config.get("setup", {})
    setup_script = setup.get("script", "")
    if setup_script:
        setup_script = substitute_variables(setup_script, variables)
        try:
            subprocess.run(
                setup_script,
                shell=True,
                check=True,
                cwd=work_dir,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            return TestSuiteResult(
                name=suite_name,
                container=container_name,
                total=0,
                failed=1,
                results=[TestResult(
                    name="Setup",
                    passed=False,
                    duration=0,
                    message=f"Setup failed: {e.stderr.decode() if e.stderr else str(e)}",
                )],
            )

    # Container health check
    health_result = _run_container_health_check(container_path, work_dir, variables)
    if not health_result.passed:
        # Get and filter tests to know how many to skip
        tests = config.get("tests", [])
        if test_filter:
            pattern = re.compile(test_filter, re.IGNORECASE)
            tests = [t for t in tests if pattern.search(t.get("name", ""))]

        skip_results = [health_result]
        for test in tests:
            skip_result = TestResult(
                name=test.get("name", "Unnamed test"),
                passed=False,
                duration=0,
                start_time=health_result.start_time,
                message="Skipped: container health check failed",
            )
            skip_results.append(skip_result)

        # Report results via callback/queue so they appear in JSONL output
        for r in skip_results:
            if on_test_complete is not None:
                on_test_complete(suite_name, container_name, r)
            if result_queue is not None:
                result_queue.put({
                    "suite": suite_name,
                    "container": container_name,
                    "test": r.name,
                    "passed": r.passed,
                    "start_time": r.start_time,
                    "duration": r.duration,
                    "message": r.message,
                    "exit_code": r.exit_code,
                    "stdout": r.stdout,
                    "stderr": r.stderr,
                })

        return TestSuiteResult(
            name=suite_name,
            container=container_name,
            total=len(skip_results),
            failed=len(skip_results),
            results=skip_results,
        )

    # Get and filter tests
    tests = config.get("tests", [])
    if test_filter:
        pattern = re.compile(test_filter, re.IGNORECASE)
        tests = [t for t in tests if pattern.search(t.get("name", ""))]

    # Run tests
    results = []
    for test in tests:
        test_name = test.get("name", "Unnamed test")
        test_key = f"{suite_name}: {test_name}"

        # Track running test
        if running_tests is not None:
            running_tests[test_key] = True

        result = run_single_test(
            test=test,
            container_path=container_path,
            variables=variables,
            work_dir=work_dir,
            global_env_setup=global_env_setup,
            default_timeout=default_timeout,
        )
        results.append(result)

        # Remove from running tests
        if running_tests is not None:
            running_tests.pop(test_key, None)

        # Call callback immediately after each test (for sequential mode)
        if on_test_complete is not None:
            on_test_complete(suite_name, container_name, result)

        # Put result on queue (for parallel mode)
        if result_queue is not None:
            result_queue.put({
                "suite": suite_name,
                "container": container_name,
                "test": result.name,
                "passed": result.passed,
                "start_time": result.start_time,
                "duration": result.duration,
                "message": result.message,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            })

        if verbose:
            status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
            console.print(f"  {status} {result.name} ({result.duration:.2f}s)")
            if not result.passed:
                console.print(f"    [dim]{result.message}[/]")

    # Run cleanup script
    cleanup = config.get("cleanup", {})
    cleanup_script = cleanup.get("script", "")
    if cleanup_script:
        cleanup_script = substitute_variables(cleanup_script, variables)
        try:
            subprocess.run(
                cleanup_script,
                shell=True,
                cwd=work_dir,
                capture_output=True,
            )
        except Exception:
            pass  # Ignore cleanup errors

    duration = time.time() - start_time
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    return TestSuiteResult(
        name=suite_name,
        container=container_name,
        total=len(results),
        passed=passed,
        failed=failed,
        duration=duration,
        results=results,
    )


def run_test_suite_wrapper(args: tuple) -> TestSuiteResult:
    """Wrapper for parallel execution."""
    yaml_path, containers_dir, work_dir, test_filter, verbose, result_queue, running_tests = args
    return run_test_suite(
        yaml_path, containers_dir, work_dir, test_filter, verbose,
        on_test_complete=None, result_queue=result_queue, running_tests=running_tests
    )


def prepare_tests_from_yaml(
    yaml_path: Path,
    containers_dir: Path,
    work_dir: Path,
    test_filter: str | None = None,
) -> tuple[list[PreparedTest], str | None]:
    """
    Prepare all tests from a YAML file for execution.
    Returns (list of prepared tests, error message if any).
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    suite_name = config.get("name", yaml_path.stem)
    container_name = config.get("container", "")
    default_timeout = config.get("default_timeout", 120)

    # Find container
    container_path = find_container(container_name, containers_dir)
    if not container_path:
        return [], f"Container not found: {container_name}"

    # Build variables dict
    variables = {}
    test_data = config.get("test_data", {})
    for key, value in test_data.items():
        if key == "output_dir":
            variables[key] = str(work_dir / value)
        else:
            path = Path(value)
            if not path.is_absolute():
                path = work_dir / value
            variables[key] = str(path)

    # Get global env setup
    global_env_setup = config.get("env_setup")

    # Clean output directory before running suite
    if "output_dir" in variables:
        output_dir = Path(variables["output_dir"])
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run setup script
    setup = config.get("setup", {})
    setup_script = setup.get("script", "")
    if setup_script:
        setup_script = substitute_variables(setup_script, variables)
        try:
            subprocess.run(
                setup_script,
                shell=True,
                check=True,
                cwd=work_dir,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            return [], f"Setup failed: {e.stderr.decode() if e.stderr else str(e)}"

    # Container health check
    health_result = _run_container_health_check(container_path, work_dir, variables)
    if not health_result.passed:
        return [], f"Container health check failed: {health_result.message}"

    # Get and filter tests
    tests = config.get("tests", [])
    if test_filter:
        pattern = re.compile(test_filter, re.IGNORECASE)
        tests = [t for t in tests if pattern.search(t.get("name", ""))]

    # Create prepared tests
    prepared = []
    for test in tests:
        prepared.append(PreparedTest(
            suite_name=suite_name,
            container_name=container_name,
            container_path=container_path,
            test=test,
            variables=variables,
            work_dir=work_dir,
            global_env_setup=global_env_setup,
            default_timeout=default_timeout,
        ))

    return prepared, None


def run_prepared_test_wrapper(args: tuple) -> tuple[str, str, TestResult]:
    """Wrapper for running a single prepared test in parallel."""
    prepared, running_tests = args

    test_name = prepared.test.get("name", "Unnamed test")
    test_key = f"{prepared.suite_name}: {test_name}"

    # Track running test
    if running_tests is not None:
        running_tests[test_key] = True

    result = run_single_test(
        test=prepared.test,
        container_path=prepared.container_path,
        variables=prepared.variables,
        work_dir=prepared.work_dir,
        global_env_setup=prepared.global_env_setup,
        default_timeout=prepared.default_timeout,
    )

    # Remove from running tests
    if running_tests is not None:
        running_tests.pop(test_key, None)

    return prepared.suite_name, prepared.container_name, result


def main():
    parser = argparse.ArgumentParser(
        description="Run neurocontainer tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "yaml_files",
        nargs="*",
        help="YAML test files to run (default: all *.yaml files)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "-c", "--containers-dir",
        type=Path,
        default=Path("containers"),
        help="Directory containing container files (default: containers)",
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        help="Filter tests by name pattern (regex)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Hide individual test results (only show summary)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available test files",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only show failed tests in output",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Write results to JSON file",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("test_results.log"),
        help="Write detailed test log (default: test_results.log)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log file output",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("test_results.jsonl"),
        help="Write streaming results to JSONL file (default: test_results.jsonl)",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Disable JSONL streaming output",
    )

    args = parser.parse_args()

    base_dir = Path.cwd()
    tests_dir = base_dir / "tests"
    work_dir = base_dir / "work"
    containers_dir = args.containers_dir.resolve()

    # Ensure work directory exists
    work_dir.mkdir(exist_ok=True)

    # Find YAML files in tests/ directory
    if args.yaml_files:
        yaml_files = []
        for pattern in args.yaml_files:
            # Check tests/ directory first
            yaml_files.extend(tests_dir.glob(pattern))
            # Also check if absolute/relative path was given
            if Path(pattern).exists():
                yaml_files.append(Path(pattern))
    else:
        yaml_files = list(tests_dir.glob("*.yaml"))

    yaml_files = sorted(set(yaml_files))

    if args.list:
        console.print(Panel(f"[bold]Available Test Files[/] (in tests/)", box=box.ROUNDED))
        for f in yaml_files:
            console.print(f"  {f.name}")
        console.print(f"\n[dim]Total: {len(yaml_files)} files[/]")
        return 0

    if not yaml_files:
        console.print(f"[red]No YAML test files found in {tests_dir}[/]")
        return 1

    console.print(Panel(
        f"[bold]Neurocontainer Test Runner[/]\n"
        f"Files: {len(yaml_files)} | Workers: {args.jobs} | Filter: {args.filter or 'none'}\n"
        f"Tests dir: {tests_dir} | Work dir: {work_dir}",
        box=box.ROUNDED,
    ))

    all_results: list[TestSuiteResult] = []
    start_time = time.time()

    # Open JSONL file for streaming results
    jsonl_file = None
    if not args.no_jsonl:
        jsonl_file = open(args.jsonl, "w")

    # Lock for thread-safe JSONL writing
    jsonl_lock = threading.Lock()

    def write_jsonl_record(record: dict):
        """Write a single record to JSONL file (thread-safe)."""
        if jsonl_file is None:
            return
        with jsonl_lock:
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()

    def write_test_result_callback(suite_name: str, container: str, test: TestResult):
        """Callback for sequential mode to write results immediately."""
        write_jsonl_record({
            "suite": suite_name,
            "container": container,
            "test": test.name,
            "passed": test.passed,
            "start_time": test.start_time,
            "duration": test.duration,
            "message": test.message,
            "exit_code": test.exit_code,
            "stdout": test.stdout,
            "stderr": test.stderr,
        })

    if args.jobs > 1:
        # Prepare all tests from all YAML files
        console.print("[dim]Preparing tests...[/]")
        all_prepared_tests: list[PreparedTest] = []
        suite_errors: dict[str, str] = {}

        for yaml_path in yaml_files:
            try:
                prepared, error = prepare_tests_from_yaml(
                    yaml_path, containers_dir, work_dir, args.filter
                )
                if error:
                    suite_errors[yaml_path.stem] = error
                else:
                    all_prepared_tests.extend(prepared)
            except Exception as e:
                suite_errors[yaml_path.stem] = str(e)

        # Shuffle tests to spread load across containers
        random.shuffle(all_prepared_tests)
        total_tests = len(all_prepared_tests)

        console.print(f"[dim]Prepared {total_tests} tests from {len(yaml_files)} suites (shuffled)[/]")

        # Report any suite errors
        for suite_name, error in suite_errors.items():
            console.print(f"[red]✗[/] {suite_name}: {error}")

        # Parallel execution
        manager = Manager()
        running_tests = manager.dict()
        test_counts = manager.dict()
        test_counts["passed"] = 0
        test_counts["failed"] = 0
        test_counts["completed"] = 0

        # Collect results by suite for aggregation
        suite_results: dict[str, list[TestResult]] = {}
        suite_containers: dict[str, str] = {}
        results_lock = threading.Lock()

        # Background thread to update progress with running tests
        progress_stop_event = threading.Event()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[green]{task.fields[passed]}[/] passed | [red]{task.fields[failed]}[/] failed | {task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task("Running tests...", total=total_tests, passed=0, failed=0)

            def update_running_description():
                """Update progress description with currently running tests."""
                while not progress_stop_event.is_set():
                    try:
                        # Update counts from shared dict
                        passed = test_counts["passed"]
                        failed = test_counts["failed"]
                        completed = test_counts["completed"]
                        progress.update(task, completed=completed, passed=passed, failed=failed)

                        running = list(running_tests.keys())
                        if running:
                            # Show up to 3 running tests
                            display = running[:3]
                            if len(running) > 3:
                                desc = f"Running: {', '.join(display)} (+{len(running)-3} more)"
                            else:
                                desc = f"Running: {', '.join(display)}"
                        else:
                            desc = "Running tests..."
                        progress.update(task, description=desc)
                    except Exception:
                        pass
                    time.sleep(0.25)

            desc_thread = threading.Thread(target=update_running_description, daemon=True)
            desc_thread.start()

            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {
                    executor.submit(
                        run_prepared_test_wrapper,
                        (prepared, running_tests),
                    ): prepared
                    for prepared in all_prepared_tests
                }

                for future in as_completed(futures):
                    suite_name, container_name, result = future.result()

                    # Update counts
                    test_counts["completed"] = test_counts["completed"] + 1
                    if result.passed:
                        test_counts["passed"] = test_counts["passed"] + 1
                    else:
                        test_counts["failed"] = test_counts["failed"] + 1

                    # Store result for suite aggregation
                    with results_lock:
                        if suite_name not in suite_results:
                            suite_results[suite_name] = []
                            suite_containers[suite_name] = container_name
                        suite_results[suite_name].append(result)

                    # Write to JSONL
                    write_jsonl_record({
                        "suite": suite_name,
                        "container": container_name,
                        "test": result.name,
                        "passed": result.passed,
                        "start_time": result.start_time,
                        "duration": result.duration,
                        "message": result.message,
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    })

                    # Show individual test result if not quiet
                    if not args.quiet:
                        test_status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
                        progress.console.print(f"  {test_status} {suite_name}: {result.name} ({result.duration:.2f}s)")
                        if not result.passed:
                            progress.console.print(f"    [dim]{result.message}[/]")

            # Stop background threads
            progress_stop_event.set()
            desc_thread.join(timeout=1.0)

        # Aggregate results into TestSuiteResult objects
        for suite_name, results in suite_results.items():
            passed = sum(1 for r in results if r.passed)
            failed = sum(1 for r in results if not r.passed)
            duration = sum(r.duration for r in results)
            all_results.append(TestSuiteResult(
                name=suite_name,
                container=suite_containers[suite_name],
                total=len(results),
                passed=passed,
                failed=failed,
                duration=duration,
                results=results,
            ))

        # Add error results for suites that failed to prepare
        for suite_name, error in suite_errors.items():
            all_results.append(TestSuiteResult(
                name=suite_name,
                container="",
                total=0,
                failed=1,
                results=[TestResult(
                    name="Suite preparation",
                    passed=False,
                    duration=0,
                    message=error,
                )],
            ))
    else:
        # Sequential execution
        for yaml_path in yaml_files:
            console.print(f"\n[bold cyan]Running: {yaml_path.name}[/]")
            result = run_test_suite(
                yaml_path,
                containers_dir,
                work_dir,
                args.filter,
                verbose=not args.quiet,
                on_test_complete=write_test_result_callback,
            )
            all_results.append(result)

            status = "[green]PASS[/]" if result.failed == 0 else "[red]FAIL[/]"
            console.print(f"  {status} {result.passed}/{result.total} tests passed ({result.duration:.1f}s)")

    # Close JSONL file
    if jsonl_file is not None:
        jsonl_file.close()
        console.print(f"[dim]Streaming results written to {args.jsonl}[/]")

    total_duration = time.time() - start_time

    # Summary
    console.print("\n")

    total_tests = sum(r.total for r in all_results)
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    suites_passed = sum(1 for r in all_results if r.failed == 0)
    suites_failed = sum(1 for r in all_results if r.failed > 0)

    # Results table
    table = Table(title="Test Results Summary", box=box.ROUNDED)
    table.add_column("Suite", style="cyan")
    table.add_column("Passed", style="green", justify="right")
    table.add_column("Failed", style="red", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Status")

    for result in sorted(all_results, key=lambda r: (-r.failed, r.name)):
        if args.failed_only and result.failed == 0:
            continue

        status = "[green]PASS[/]" if result.failed == 0 else "[red]FAIL[/]"
        table.add_row(
            result.name,
            str(result.passed),
            str(result.failed),
            str(result.total),
            f"{result.duration:.1f}s",
            status,
        )

    console.print(table)

    # Show failed tests details
    if total_failed > 0:
        console.print("\n[bold red]Failed Tests:[/]")
        for result in all_results:
            for test in result.results:
                if not test.passed:
                    console.print(f"  [red]✗[/] {result.name} > {test.name}")
                    console.print(f"    [dim]{test.message}[/]")

    # Final summary
    console.print(Panel(
        f"[bold]Final Summary[/]\n\n"
        f"Suites: [green]{suites_passed} passed[/], [red]{suites_failed} failed[/] "
        f"({len(all_results)} total)\n"
        f"Tests:  [green]{total_passed} passed[/], [red]{total_failed} failed[/] "
        f"({total_tests} total)\n"
        f"Time:   {total_duration:.1f}s",
        box=box.ROUNDED,
    ))

    # Write JSON output if requested
    if args.output:
        from datetime import datetime

        output_data = {
            "summary": {
                "total_suites": len(all_results),
                "suites_passed": suites_passed,
                "suites_failed": suites_failed,
                "total_tests": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_failed,
                "duration": total_duration,
                "run_timestamp": datetime.now().isoformat(),
            },
            "suites": [
                {
                    "name": r.name,
                    "container": r.container,
                    "total": r.total,
                    "passed": r.passed,
                    "failed": r.failed,
                    "duration": r.duration,
                    "tests": [
                        {
                            "name": t.name,
                            "passed": t.passed,
                            "start_time": t.start_time,
                            "duration": t.duration,
                            "message": t.message,
                        }
                        for t in r.results
                    ],
                }
                for r in all_results
            ],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"\n[dim]Results written to {args.output}[/]")

    # Write log file
    if not args.no_log:
        from datetime import datetime

        with open(args.log, "w") as f:
            f.write(f"# Neurocontainer Test Results\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total Duration: {total_duration:.2f}s\n")
            f.write(f"#\n")
            f.write(f"# Format: STATE | START_TIME | DURATION | SUITE | TEST_NAME | MESSAGE\n")
            f.write(f"#\n\n")

            for suite_result in sorted(all_results, key=lambda r: r.name):
                for test in suite_result.results:
                    state = "PASS" if test.passed else "FAIL"
                    f.write(
                        f"{state} | {test.start_time} | {test.duration:.3f}s | "
                        f"{suite_result.name} | {test.name} | {test.message}\n"
                    )

        console.print(f"[dim]Log written to {args.log}[/]")

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
