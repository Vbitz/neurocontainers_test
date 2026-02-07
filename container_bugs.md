# Genuine Container Bugs Report

Analysis of 122 JSONL result files. Only issues confirmed to be problems with the container image itself are included. Test suite issues, cascade failures, missing test data, shell quoting errors, and apptainer runtime issues are excluded.

---

## 1. Containers with Missing Binaries

### mitkdiffusion
**Severity: Critical -- 0/128 tests pass**

All 23 MITK Diffusion wrapper scripts are missing from the container root `/`. The container ships no functional tool entry points.

Missing scripts include:
- `/MitkStreamlineTractography.sh`
- `/MitkTractDensity.sh`
- `/MitkFiberProcessing.sh`
- `/MitkTensorReconstruction.sh`
- `/MitkDiffusionIndices.sh`
- `/MitkFiberJoin.sh`
- ... and 17 others

Evidence:
```
exit=127 stdout: /MitkTractDensity.sh: No such file or directory
exit=127 stdout: /MitkStreamlineTractography.sh: No such file or directory
```

### bidsappaa
**Severity: High -- 27/97 tests pass**

Both FSL and FreeSurfer installations are broken:

**FSL**: Shell scripts reference hardcoded paths to helper binaries that don't exist at those locations. The bet script (and others) call `/bin/remove_ext`, `/bin/imtest`, `/bin/fslhd`, `/bin/bet2`, `/bin/fslstats`, `/bin/fslval` -- but these binaries are not at `/bin/`.
```
/opt/fsl/bin/bet: 1: /opt/fsl/bin/bet: /bin/remove_ext: not found
/opt/fsl/bin/bet: 1: /opt/fsl/bin/bet: /bin/imtest: not found
/opt/fsl/bin/bet: 236: /opt/fsl/bin/bet: /bin/bet2: not found
/opt/fsl/bin/fslinfo: 77: /opt/fsl/bin/fslinfo: /bin/fslhd: not found
```

**FreeSurfer**: mri_convert wrapper script references a missing `/sources.sh` and the actual binary `mri_convert.bin` is not found.
```
/opt/freesurfer/bin/mri_convert: line 2: /sources.sh: No such file or directory
/opt/freesurfer/bin/mri_convert: line 3: mri_convert.bin: command not found
```

### qsirecon
**Severity: Low -- 32/125 tests pass (most failures are unrelated)**

The ANTs binary `antsMotionCorr` is missing from the expected path.
```
exit=127 stdout: /opt/ants/bin/antsMotionCorr: No such file or directory
```

### ezbids
**Severity: Low -- 55/90 tests pass**

MongoDB shell (`mongosh`) is missing from the container.
```
exit=127 stdout: mongosh: command not found
```

---

## 2. Containers with Missing Libraries/Dependencies

### hdbet
**Severity: High -- 9/34 tests pass**

The `nibabel` Python module is missing from the default Python environment. The hd-bet binary works (runs in its own environment), but any direct Python `import nibabel` fails. 9 tests fail with this error.
```
ModuleNotFoundError: No module named 'nibabel'
```

### mrsiproc
**Severity: High -- 80/107 tests pass**

The MATLAB Compiled Runtime shared library `libmwlaunchermain.so` is missing. All 7 compiled MATLAB binaries in the pipeline fail to load.
```
/opt/mrsi_pipeline_neurodesk/matlab_compiled/CreateSpectralNiftiMap: error while loading shared libraries:
  libmwlaunchermain.so: cannot open shared object file: No such file or directory
```
Affected binaries: `CreateSpectralNiftiMap`, `extract_met_maps`, `segmentation_simple`, `GetPar_CreateTempl_MaskPart1`, `julia_write_lcm_files`, `MRSI_Reconstruction`, `extract_spectra`.

### tractseg
**Severity: Moderate -- 47/119 tests pass**

The `libfftw3.so.3` shared library is missing. The MRtrix3 `mrfilter` command fails to load. 4 tests fail with this error.
```
mrfilter: error while loading shared libraries: libfftw3.so.3: cannot open shared object file:
  No such file or directory
```

### ezbids
**Severity: High (Python environment corruption)**

The Python standard library `math` module cannot be imported, indicating a corrupted Python environment. FSL's `tmpnam` script triggers the error chain: `tempfile` -> `random` -> `math` (fails). 3 tests fail with this.
```
File "/usr/lib/python3.8/random.py", line 41, in <module>
    from math import log as _log, exp as _exp, pi as _pi, e as _e, ceil as _ceil
ModuleNotFoundError: No module named 'math'
```

### nipype
**Severity: Moderate -- 33/82 tests pass**

The `pybids` (`bids`) Python module is missing. nipype's BIDS I/O interface cannot initialize. 1 test directly fails; many others cascade.
```
File "/opt/miniconda-latest/lib/python3.9/site-packages/nipype/interfaces/io.py", line 2952
    from bids import layout as bidslayout
ModuleNotFoundError: No module named 'bids'
```

### rabies
**Severity: Low -- 59/78 tests pass**

The `nilearn.maskers` submodule is missing (requires nilearn >= 0.9.0; container has an older version). 2 tests fail.
```
ModuleNotFoundError: No module named 'nilearn.maskers'
```

### fastsurfer
**Severity: Moderate -- 58/82 tests pass**

12 tests fail due to two related container packaging issues:

**PYTHONPATH not configured**: FastSurfer's internal modules (`map_surf_label`, `image_io`, `align_points`, `smooth_aparc`, `create_annotation`) cannot be imported because `/fastsurfer/recon_surf/` is not on PYTHONPATH. 5 tests fail.
```
File "/fastsurfer/recon_surf/N4_bias_correct.py", line 27
    import image_io as iio
ModuleNotFoundError: No module named 'image_io'
```

**Changed internal APIs**: Several modules no longer export expected functions (`main` function removed from `rewrite_mc_surface`, `paint_cc_into_pred`, `map_surf_label`, `rewrite_oriented_surface`, `spherically_project`; `__version__` removed from `FastSurferCNN.version`; `LTA` class removed from `recon_surf.lta`). 7 tests fail.
```
ImportError: cannot import name 'main' from 'recon_surf.rewrite_mc_surface'
ImportError: cannot import name '__version__' from 'FastSurferCNN.version'
```

### mritools
**Severity: Moderate -- 10/54 tests pass (most failures are unrelated)**

The Julia package `MriResearchTools` is not installed despite being the container's primary tool.
```
ERROR: ArgumentError: Package MriResearchTools not found in current path:
- Run `import Pkg; Pkg.add("MriResearchTools")` to install the MriResearchTools package.
```

### pcntoolkit
**Severity: Low -- 89/97 tests pass**

The `g++` compiler is missing, preventing PyTensor from compiling C implementations. PyTensor falls back to pure Python, causing severe performance degradation. This is a packaging oversight.
```
WARNING (pytensor.configdefaults): g++ not detected! PyTensor will be unable to compile
  C-implementations and will default to Python. Performance may be severely degraded.
```

---

## 3. Containers with License Issues

### matlab
**Severity: Critical -- 2/136 tests pass**

The MATLAB container has no license file configured. All MATLAB-dependent tests (132) fail with license checkout errors. The container is unusable without external license configuration.
```
License checkout failed.
License Manager Error -1
The license file cannot be found.
Feature: MATLAB
License path: /home/astra/Downloads/*...
```

---

## 4. Containers with Tool Crashes on Valid Input

### qupath
**Severity: Moderate -- 75/120 tests pass**

QuPath segfaults (exit 139) on all `script --cmd` invocations, even on trivial Groovy expressions like `Math.sin(Math.PI/2)`. 37 tests fail. The likely cause is that JavaFX fails to initialize in headless mode, and QuPath's script runner does not handle this gracefully.
```
Segmentation fault (core dumped) QuPath script --cmd "println('sin(pi/2) = ' + Math.sin(Math.PI/2))"
Segmentation fault (core dumped) QuPath script --cmd "import qupath.lib.roi.ROIs; ..."
Segmentation fault (core dumped) QuPath script --cmd "import qupath.lib.common.GeneralTools; ..."
```
Non-script QuPath commands (like `convert-ome`) work correctly.

---

## 5. Containers with Broken Tool APIs

### vesselapp
**Severity: Moderate -- 63/74 tests pass**

7 function classes have a broken `__exec__` method due to Python name mangling. The parent class `Function` defines `__exec__()` but subclasses cannot access it because Python mangles double-underscore names to `_ClassName__exec__`. Affected classes: `SimpleThresholding`, `NCWHDTensorInterpolation`, `WHDTensor2NCWHD`, `N1WHDTensor2NCWHDOnehot`, `NumpyArray2TorchTensor`, `RemoveSmallObjectsFromBinaryArray`.
```
AttributeError: 'SimpleThresholding' object has no attribute '_Function__exec__'
AttributeError: 'NCWHDTensorInterpolation' object has no attribute '_Function__exec__'
```

### amico
**Severity: Moderate -- 62/85 tests pass**

Multiple model classes have changed function signatures. Methods like `get_signal()` accept fewer positional arguments than expected, and object attributes (`nS`, `isExvivo`) no longer exist. 10 tests fail.
```
TypeError: Stick.get_signal() takes 2 positional arguments but 3 were given
TypeError: Zeppelin.get_signal() takes 3 positional arguments but 4 were given
AttributeError: 'CylinderZeppelinBall' object has no attribute 'isExvivo'
AttributeError: 'str' object has no attribute 'nS'
```

### halfpipe
**Severity: Moderate -- 36/60 tests pass**

3 tests fail because internal module paths have changed. `Setting`, `ICAAROMASetting`, and the `preprocessing` submodule of `halfpipe.workflows` no longer exist at their expected import paths.
```
ImportError: cannot import name 'Setting' from 'halfpipe.model.setting'
ImportError: cannot import name 'ICAAROMASetting' from 'halfpipe.model.setting'
ImportError: cannot import name 'preprocessing' from 'halfpipe.workflows'
```

### dcm2bids
**Severity: Low -- 47/62 tests pass**

2 tests fail due to API restructuring. `bids_version` was removed from `dcm2bids.version`, and the `dcm2bids.helper` submodule no longer exists.
```
ImportError: cannot import name 'bids_version' from 'dcm2bids.version'
ModuleNotFoundError: No module named 'dcm2bids.helper'
```

### glmsingle
**Severity: Low -- 72/77 tests pass**

The function `olsmatrix2` is missing from `glmsingle.ols.olsmatrix`. 1 test fails.
```
ImportError: cannot import name 'olsmatrix2' from 'glmsingle.ols.olsmatrix'
```

### deepretinotopy
**Severity: Low -- 107/136 tests pass**

The `Model` class is missing from `utils.model`. 1 test fails.
```
ImportError: cannot import name 'Model' from 'utils.model'
  (/opt/deepRetinotopy_TheToolbox/utils/model.py)
```

### bidsapppymvpa
**Severity: Low -- 90/97 tests pass**

`EventRelatedDataset` cannot be imported from PyMVPA (likely renamed in installed version). 1 test fails.
```
ImportError: cannot import name EventRelatedDataset
```

---

## Summary Table

| Container | Bug Type | Severity | Pass Rate | # Affected Tests |
|-----------|----------|----------|-----------|-----------------|
| mitkdiffusion | Missing binaries (all wrappers) | Critical | 0/128 | 128 |
| matlab | License not configured | Critical | 2/136 | 132 |
| bidsappaa | Broken FSL + FreeSurfer install | High | 27/97 | 15+ |
| hdbet | Missing nibabel module | High | 9/34 | 9 |
| mrsiproc | Missing MATLAB runtime lib | High | 80/107 | 7 |
| ezbids | Corrupted Python env + missing mongosh | High | 55/90 | 4 |
| fastsurfer | PYTHONPATH + changed APIs | Moderate | 58/82 | 12 |
| nipype | Missing pybids module | Moderate | 33/82 | 1+ |
| tractseg | Missing libfftw3.so.3 | Moderate | 47/119 | 4 |
| amico | Changed API signatures | Moderate | 62/85 | 10 |
| vesselapp | Broken __exec__ method | Moderate | 63/74 | 7 |
| halfpipe | Changed import paths | Moderate | 36/60 | 3 |
| qupath | Segfault in script mode | Moderate | 75/120 | 37 |
| mritools | Missing Julia package | Moderate | 10/54 | 1 |
| qsirecon | Missing antsMotionCorr | Low | 32/125 | 1 |
| dcm2bids | Changed API structure | Low | 47/62 | 2 |
| rabies | Outdated nilearn | Low | 59/78 | 2 |
| glmsingle | Missing function | Low | 72/77 | 1 |
| deepretinotopy | Missing Model class | Low | 107/136 | 1 |
| pcntoolkit | Missing g++ compiler | Low | 89/97 | 0 (perf only) |
| bidsapppymvpa | Renamed class | Low | 90/97 | 1 |
