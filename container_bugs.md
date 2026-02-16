# Confirmed Container Bugs

Analysis of 122 JSONL result files. Only issues confirmed to be problems with the
container image itself are listed. Test suite bugs, cascade failures, missing test
data, and version mismatches between tests and installed software are excluded.

---

## 1. Missing Binaries / Broken Installations

### bidsappaa
**Severity: Moderate -- 97/97 tests pass with workarounds**

FSL and FreeSurfer environment variables are not configured in the container.
FSL works after sourcing `/opt/fsl/etc/fslconf/fsl.sh` and adding `/opt/fsl/bin`
to PATH. FreeSurfer works after setting `FREESURFER_HOME=/opt/freesurfer` and
adding `/opt/freesurfer/bin` to PATH, but `mri_convert` operations fail because
no FreeSurfer license file is included. The compiled MATLAB AA application also
fails due to corrupted/empty `.m` toolbox files.

**Why this is a container bug:** The container should have FSL/FreeSurfer
environment configured at build time. The missing FreeSurfer license and
corrupted MATLAB Runtime are build-time omissions.

```
# Without env setup:
/opt/fsl/bin/bet: 1: /bin/remove_ext: not found
# mri_convert conversion:
ERROR: FreeSurfer license file not found (no license.txt)
# AA MATLAB:
The file "/opt/automaticanalysis5/.../matlabrc.m" cannot be executed. File is empty
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
**Severity: Critical -- 3/120 tests pass (only error-handling tests)**

QuPath segfaults (exit 139) on ALL invocations, including `--version`,
`--help`, `script`, and `convert-ome`. The binary is completely non-functional.

**Why this is a container bug:** The QuPath ELF launcher binary segfaults
immediately before any Java code executes. This appears to be a shared library
or runtime incompatibility. The container cannot perform any of its functions.

```
QuPath --version -> Segmentation fault (core dumped) (exit 139)
QuPath --help -> Segmentation fault (core dumped) (exit 139)
QuPath script --cmd "println('hello')" -> Segmentation fault (core dumped) (exit 139)
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

### cat12
**Severity: High -- 34/50 tests pass**

All CAT12 MEX files require GLIBC_2.29 but the container's base OS only provides
an older glibc. The segmentation pipeline fails at the first MEX call (SANLM
denoising) and all subsequent steps also fail since they use the same
incompatible MEX files. 16 tests depend on segmentation output.

**Why this is a container bug:** The MEX files (`cat_sanlm.mexa64`,
`cat_amap.mexa64`, etc.) were compiled against glibc 2.29+ but packaged into a
container with an older glibc. The MCR (v93/R2017b) loads but cannot execute
any CAT12-specific compiled code.

```
Invalid MEX-file 'cat_sanlm.mexa64': 'cat_sanlm.mexa64' is not a valid shared library.
ldd: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found
```

### ants
**Severity: Low -- 2 tools affected**

`ResetDirection` segfaults (exit 139) on valid input. `ImageMath TimeSeriesDisassemble`
aborts (exit 134) with `std::out_of_range` when processing 4D data.

**Why this is a container bug:** Both tools are provided by the ANTs 2.6.0 container
and crash on valid inputs. `ResetDirection` segfaults when given a valid NIfTI image.
`TimeSeriesDisassemble` crashes with a C++ string bounds error when the output prefix
doesn't contain a file extension.

```
ResetDirection 3 input.nii.gz output.nii.gz -> Segmentation fault (exit 139)
ImageMath 4 prefix_ TimeSeriesDisassemble input.nii.gz -> std::out_of_range (exit 134)
```

### aslprep
**Severity: Low -- 2 BET options affected**

`bet -R` (robust) fails because `dc` (desk calculator) is not installed. `bet -B`
(bias field cleanup) fails because `standard_space_roi` has cascading internal failures.
Basic `bet` works fine.

**Why this is a container bug:** `bet -R` is a standard FSL BET option that requires the
`dc` binary for center-of-gravity calculation. The container has FSL installed but omitted
this dependency. `bet -B` calls `standard_space_roi` internally which also fails with
SIGABRT, indicating incomplete FSL installation.

```
/opt/conda/envs/aslprep/bin/bet: line 265: dc: command not found
standard_space_roi: Aborted (core dumped)
```

### spinalcordtoolbox
**Severity: Critical -- 0/98 tests pass (container image unreadable)**

The container image file is corrupted. Apptainer cannot mount the squashfs
image. No tests can run.

**Why this is a container bug:** The `.simg` file is physically corrupted —
apptainer's squashfuse driver cannot read the squashfs filesystem.

```
FATAL: container creation failed: image driver mount failure:
  image driver squashfuse_ll instance exited with error:
  squashfuse_ll exited: Something went wrong trying to read the squashfs image.
```

---

## Summary

| Container | Bug Type | Severity |
|-----------|----------|----------|
| bidsappaa | Missing env config, FreeSurfer license, corrupted MATLAB | Moderate |
| mrsiproc | Missing MATLAB runtime .so | High |
| ezbids | Corrupted Python stdlib | High |
| tractseg | Missing libfftw3.so.3 | Moderate |
| mritools | Primary Julia package not installed | Moderate |
| vesselapp | Python name-mangling code bug | Moderate |
| qupath | ALL invocations segfault (binary non-functional) | Critical |
| bart | SIGABRT on missing input files | Low |
| niimath | SIGABRT on missing input files | Low |
| niftyreg | SIGSEGV on missing input files | Low |
| dsistudio | Missing CUDA drivers for registration | Low |
| cat12 | GLIBC_2.29 mismatch breaks all MEX files | High |
| ants | ResetDirection segfault, TimeSeriesDisassemble crash | Low |
| aslprep | Missing dc binary, broken standard_space_roi | Low |
| spinalcordtoolbox | Corrupted container image (squashfs unreadable) | Critical |
