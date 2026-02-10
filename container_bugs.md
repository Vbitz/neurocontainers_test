# Confirmed Container Bugs

Analysis of 122 JSONL result files. Only issues confirmed to be problems with the
container image itself are listed. Test suite bugs, cascade failures, missing test
data, and version mismatches between tests and installed software are excluded.

---

## 1. Missing Binaries / Broken Installations

### bidsappaa
**Severity: High -- 27/97 tests pass**

Both FSL and FreeSurfer installations have broken internal script paths.

**Why this is a container bug:** The errors come from *inside* the tool scripts
themselves, not from the test commands. FSL's own `bet` script calls
`/bin/remove_ext` and `/bin/bet2` — these are FSL utilities that should be on
the container's PATH or symlinked from `/bin/`. The scripts were installed
incorrectly during container build. Similarly, FreeSurfer's `mri_convert` wrapper
tries to source `/sources.sh` which doesn't exist, then can't find
`mri_convert.bin`.

```
/opt/fsl/bin/bet: 1: /bin/remove_ext: not found
/opt/fsl/bin/bet: 236: /bin/bet2: not found
/opt/freesurfer/bin/mri_convert: line 2: /sources.sh: No such file or directory
/opt/freesurfer/bin/mri_convert: line 3: mri_convert.bin: command not found
```

### mrsiproc
**Severity: High -- 80/107 tests pass**

The MATLAB Compiled Runtime shared library `libmwlaunchermain.so` is missing.
All 7 compiled MATLAB binaries in the pipeline fail to load.

**Why this is a container bug:** These are the container's own compiled binaries
dynamically linking to a .so that should have been included in the container.
The MCR was partially installed — the binaries exist but their required runtime
library is absent. This is a build-time dependency that was missed.

```
/opt/mrsi_pipeline_neurodesk/matlab_compiled/CreateSpectralNiftiMap:
  error while loading shared libraries: libmwlaunchermain.so:
  cannot open shared object file: No such file or directory
```

Affected: `CreateSpectralNiftiMap`, `extract_met_maps`, `segmentation_simple`,
`GetPar_CreateTempl_MaskPart1`, `julia_write_lcm_files`, `MRSI_Reconstruction`,
`extract_spectra`.

### tractseg
**Severity: Moderate -- 47/119 tests pass**

`libfftw3.so.3` is missing. MRtrix3's `mrfilter` command (bundled in the
container) fails to load.

**Why this is a container bug:** `mrfilter` is a tool shipped inside this
container. It dynamically links to `libfftw3.so.3` which should have been
included as a build dependency. The binary exists but cannot execute.

```
mrfilter: error while loading shared libraries: libfftw3.so.3:
  cannot open shared object file: No such file or directory
```

### mritools
**Severity: Moderate -- 10/54 tests pass**

The Julia package `MriResearchTools` is not installed.

**Why this is a container bug:** The container is named `mritools` and its
documented purpose is to provide the `MriResearchTools` Julia package. The
Julia runtime is present but the package itself was never installed into the
depot. The container cannot perform its primary function.

```
ERROR: ArgumentError: Package MriResearchTools not found in current path
```

---

## 2. Corrupted Container Environment

### ezbids
**Severity: High -- 55/90 tests pass**

The Python standard library `math` module cannot be imported, indicating a
corrupted or stripped Python installation.

**Why this is a container bug:** The `math` module is part of Python's C
standard library (`_math.cpython-*.so`). It's required by `random`, which is
required by `tempfile`, which is used pervasively. This isn't a missing
third-party package — the Python installation itself is broken. FSL's own
scripts trigger this when they call `tmpnam`.

```
File "/usr/lib/python3.8/random.py", line 41, in <module>
    from math import log as _log, exp as _exp, pi as _pi, e as _e
ModuleNotFoundError: No module named 'math'
```

### vesselapp
**Severity: Moderate -- 63/74 tests pass**

7 function classes have a broken `__exec__` method due to Python name mangling.

**Why this is a container bug:** This is a genuine code bug in the shipped
library, not a version mismatch. The parent class `Function` defines
`__exec__()`, but Python's name mangling transforms dunder methods to
`_ClassName__exec__` per class. Subclasses calling `self.__exec__()` look for
`_SubClass__exec__` which doesn't exist. This is a bug in the source code
that was packaged into the container — it would fail in any version of Python.

```
AttributeError: 'SimpleThresholding' object has no attribute '_Function__exec__'
```

Affected: `SimpleThresholding`, `NCWHDTensorInterpolation`, `WHDTensor2NCWHD`,
`N1WHDTensor2NCWHDOnehot`, `NumpyArray2TorchTensor`, `RemoveSmallObjectsFromBinaryArray`.

### qupath
**Severity: Moderate -- 75/120 tests pass**

QuPath segfaults (exit 139) on all `script --cmd` invocations, including
trivial Groovy expressions.

**Why this is a container bug:** `QuPath script` is a documented QuPath
subcommand for headless scripting. The container should support headless
operation since it has no display. The segfault likely occurs because JavaFX
tries to initialize a display and crashes instead of falling back gracefully.
Non-script commands like `convert-ome` work fine, confirming the container
runs but this specific mode is broken.

```
Segmentation fault (core dumped) QuPath script --cmd "println('hello')"
```

---

## 3. Poor Error Handling (Crash on Missing Input)

### bart
**Severity: Low -- 5/116 tests pass (cascade failures)**

BART aborts with SIGABRT (exit 134) when input files are missing, instead of
printing an error and exiting cleanly. All 111 failing tests are cascade
failures from missing test output files (tests depend on earlier test outputs).

**Why this is a container bug:** While the test dependencies cause the missing
files, the tool should handle missing input gracefully. BART calls `abort()`
when it cannot load a CFL file, producing a core dump instead of a clean error.

```
ERROR: Loading data.
bart: fio.c:74: io_reserve: Assertion `fd >= 0' failed.
Aborted (core dumped)
```

### niimath
**Severity: Low -- 29/114 tests pass (cascade failures)**

niimath aborts with SIGABRT (exit 134) when input NIfTI files are missing.
All 82 SIGABRT failures are cascade failures from missing test output files.

**Why this is a container bug:** niimath should exit with a non-zero code and
an error message when input files don't exist, not abort with a signal.

```
** ERROR (nifti_image_read): failed to find header file for 'test_output/t1w_otsu_mask.nii.gz'
Aborted (core dumped)
```

### niftyreg
**Severity: Low -- 30/88 tests pass (cascade failures)**

NiftyReg segfaults (exit 139) when input NIfTI files are missing. 52 of 58
failures are SIGSEGV from missing test output files.

**Why this is a container bug:** NiftyReg dereferences null pointers when
nifti_image_read fails, instead of checking the return value.

```
** ERROR (nifti_image_read): failed to find header file for 'test_output/t1w_float.nii.gz'
Segmentation fault (core dumped)
```

### dsistudio
**Severity: Low -- 65/83 tests pass**

DSI Studio registration and GPU-dependent tests fail because CUDA drivers are
not available in the container environment. 8 tests fail with missing output
files because registration silently fails without GPU.

**Note:** This is an environment limitation rather than a true container bug.
The 9 help-check test failures are caused by the `env:` dict format not being
processed by the test runner (now fixed).

```
cannot obtain GPU driver and device information (CUDA ERROR 35).
Please make sure you have drivers properly installed.
```

---

## Summary

| Container | Bug Type | Severity |
|-----------|----------|----------|
| bidsappaa | Broken FSL + FreeSurfer script paths | High |
| mrsiproc | Missing MATLAB runtime .so | High |
| ezbids | Corrupted Python stdlib | High |
| tractseg | Missing libfftw3.so.3 | Moderate |
| mritools | Primary Julia package not installed | Moderate |
| vesselapp | Python name-mangling code bug | Moderate |
| qupath | Segfault in headless script mode | Moderate |
| bart | SIGABRT on missing input files | Low |
| niimath | SIGABRT on missing input files | Low |
| niftyreg | SIGSEGV on missing input files | Low |
| dsistudio | Missing CUDA drivers for registration | Low |
