# Analyse Test Results Workflow

## Goal

Classify every test result across all 116 JSONL files in `results/` into one of four categories, then produce a per-suite markdown summary with actionable next steps.

## Classification Definitions

| Category | Code | Meaning |
|----------|------|---------|
| **True Positive** | TP | Test marked `passed:true` AND it genuinely succeeded -- exit code 0, real stdout/output, command actually executed inside the container |
| **True Negative** | TN | Test marked `passed:false` AND it's a legitimate failure of the tool/container (wrong output, unsupported feature, real bug) |
| **False Positive** | FP | Test marked `passed:true` BUT it didn't actually run properly -- e.g. exit code 255 with "bash not found", shell syntax error, or stale output files from a prior run |
| **False Negative** | FN | Test marked `passed:false` BUT the failure is due to test infrastructure bugs, NOT the actual tool being broken |

## Key Failure Patterns Discovered

These are the systemic issues found during the afni analysis that likely affect many suites:

### 1. Container has no shell (`bash` not in `$PATH`)

- **Signature:** `exit_code: 255`, stderr contains `FATAL: "bash": executable file not found in $PATH`
- **Cause:** `run_tests.py` line 206 wraps every command as `apptainer exec ... bash -c '...'`. Minimal containers don't ship bash or even sh.
- **Effect:** Command never executes inside container. ALL results with this error are invalid:
  - If `passed:true` -> **FP** (nothing ran, but validation was skipped or found stale files)
  - If `passed:false` -> **FN** (reported as tool failure, but tool was never invoked)

### 2. Shell quoting breaks on parentheses in expressions

- **Signature:** `exit_code: 2`, stderr contains `syntax error near unexpected token '('`
- **Cause:** Commands with expressions like `-expr 'sin(a/100)'` have inner single quotes that break the outer `bash -c '...'` quoting. The host shell sees unmatched parens.
- **Effect:** `apptainer` is never even invoked. Same classification logic as #1.

### 3. Missing `/tmp` and `/var/tmp` in container

- **Signature:** stderr contains `WARNING: Skipping mount /tmp [tmp]: /tmp doesn't exist in container`
- **Effect:** Non-fatal warning, but tools needing temp space may fail silently or produce incorrect results.

### 4. No default exit code validation

- **Cause:** `run_tests.py` only checks exit code if `expected_exit_code` is explicitly set in the YAML. Tests using only `validate: output_exists` don't check exit codes.
- **Effect:** Tests can "pass" with exit code 255 if output files happen to exist from a prior run. These are **FP**.

### 5. Missing passwd/group files in container

- **Signature:** `WARNING: passwd file doesn't exist in container, not updating`
- **Effect:** Non-fatal, but indicates container isn't set up for user namespace mapping.

## Per-File Analysis Workflow

For each `results/<suite>.jsonl`:

### Step 1: Read the JSONL file
Parse every line. Each line is a JSON object with fields:
- `suite`, `container`, `test`, `passed`, `start_time`, `duration`, `message`, `exit_code`, `stdout`, `stderr`

### Step 2: Read the corresponding test YAML
Located at `tests/<suite>.yaml`. Understand:
- What container is being tested
- What each test command does
- What validations are expected (`expected_exit_code`, `expected_output_contains`, `validate: output_exists`)

### Step 3: Probe the container (optional, for ambiguous cases)
Run a quick sanity check:
```bash
apptainer exec containers/<container>.simg sh -c 'echo hello'
# If sh fails too, try:
apptainer exec containers/<container>.simg ls /bin/
```
This reveals whether the container has any shell at all.

### Step 4: Classify each test result

Apply this decision tree:

```
Is stderr showing "bash: executable file not found"?
  YES -> Did nothing run.
         passed:true  -> FP
         passed:false -> FN
  NO  ->
    Is stderr showing "syntax error near unexpected token"?
      YES -> Shell quoting broke on host.
             passed:true  -> FP
             passed:false -> FN
      NO  ->
        Did the command actually execute (exit_code != 255, real stdout/stderr from the tool)?
          YES ->
            passed:true  -> Likely TP (verify: exit_code 0, output makes sense)
            passed:false -> Likely TN (verify: the failure message relates to actual tool behavior)
          NO  ->
            Investigate further (timeout, container not found, other infra issue)
```

### Step 5: Write the analysis markdown

Create `results/<suite>_analysis.md` with:

1. **Header** -- container name, test date, analysis date, total test count
2. **Summary table** -- TP/TN/FP/FN counts and percentages
3. **Detailed classification table** -- every test: name, recorded pass/fail, classification, brief reason
4. **Root Causes** -- systemic issues specific to this suite
5. **Actionable Next Steps** -- concrete, prioritised fixes

## Parallelisation Strategy

- 116 JSONL files total
- Launch 10 analysis agents in parallel, each handling one file
- As each completes, launch the next until all 116 are done
- Each agent is self-contained: reads JSONL + YAML, optionally probes container, writes markdown

## Output

Each suite gets a `results/<suite>_analysis.md` file sitting next to its `results/<suite>.jsonl`.

## Common Fixes (apply across all suites)

These are the systemic fixes that would resolve the majority of issues across all suites:

| Priority | Fix | Impact |
|----------|-----|--------|
| 1 | Use `apptainer exec ... <command>` directly instead of `bash -c` wrapper | Fixes all "bash not found" failures |
| 2 | Bind-mount `/tmp` and `/var/tmp` (or use `--writable-tmpfs`) | Fixes tools needing temp space |
| 3 | Use `shlex.quote()` or write commands to temp script files | Fixes parenthesis quoting issues |
| 4 | Default `expected_exit_code` to 0 when not specified | Prevents false positives |
| 5 | Clean `test_output/` before each suite run | Prevents stale-file false positives |
| 6 | Add a container health check as the first test in every suite | Early detection of infra problems |
