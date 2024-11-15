# NUSHU - Magnes Smart Shoe for Gait-Analysis: Analysis Module Kishu

[![pipeline status](https://gitlab.com/magnes/kishu/badges/master/pipeline.svg)](https://gitlab.com/magnes/kishu/-/commits/master)

## Cloning
This repository features submodules. Make sure to clone all submodules too using
```bash
git clone --recurse-submodules git@gitlab.com:magnes/kishu.git
```

Make sure to keep the submodules up to date and at a compatible version.


## Label
This repository is part of Magnes Nushu by Magnes AG.
![Manges Nushu Label](./labels/sw-labels-REM-V3.png)


## Deploying
Deployment of the Magnes Nushu analysis suite occurs by checking out a specific
version tag on the remote machine. To do so, you first need to create a new
version tag on Git(Lab). Make sure to follow the convention outlined in the
CHANGELOG for the version number and to copy, from the CHANGELOG, the release
notes to the tag's detailed description (Release Notes).

After a new tag has been published, say `v1.0.0`, the following steps have to be
performed.
1. Connect to the remote machine via ssh (your local machine has to be\
whitelisted in order to be able to do so): `ssh debian@89.145.165.216`
2. Navigate to the `kishu` repo: `cd kishu`
3. Fetch the latest remote changes: `git fetch`
4. Checkout the latest release: `git checkout v1.0.0`
5. Make sure to have the correct cronjobs set up: `bash x/remote/install_crontab.sh`
6. Done!

_NOTE_ The version of `nushu_pryv_pyclient` has to be managed in the same way
but separately. To do so, navigate to `kishu/nushu_pryv_pyclient` and repeat
steps 3-4 for the correct `nushu_pryv_pyclient` version.


## Context
![NUSHU Overview](img/nushu/deployment-view-01.jpg)

This is the analysis module of the Magnes Nushu smart-shoe with embedded sensors
for gait analysis, i.e. REM.


## Introduction
This repository contains the source code of Magnes' gait-analysis tools. These
include data post-processing algorithms for gait-events and -parameters
extraction as well as communication with the AWS DB.
_NOTE_: The AWS DB Client is a legacy module not used in the certified MD. It is
still contained herein for development purposes. The certified version of the
code uses a Pryv database. For communication with this DB the git-submodule
`nushu_pryv_pyclient` is used.

This SW is meant to run on a remote VM. A number of executables are to be found
in `x/` and after deployment the job to be run is
```bash
  ./x/remote/run_analysis.sh
```
from within `<path_to_this_repo>/`.

For debugging/development purposes the following sub-modules can be run from `src/`:
* `run_remote_analysis.py`: be careful as this interacts with the remote DB;
* `run_local_analysis.py`: relies on locally stored data in JSON format; runs analysis on a single dataset.

Other executables which you will often have to run on your local machine when
working on this project are:
* `x/ci/run_tests.sh`
* `x/ci/run_linting.sh`
* `x/local/generate_docu.sh`
* `x/local/diagnose_coverage.sh`
* `x/local/profile_analysis.sh`


## Developing Code
Make sure to follow the guidelines reported in the `CONTRIBUTING` file. Go
read it **before** starting to do anything else in this repository.

### Testing
Make sure to always write and run the tests before committing and pushing! Tests
are to be added for each new feature/test case to `src/test/` and have to be run
with
```bash
  ./x/ci/run_tests.sh
```

Make use of `pylint` as code-linter to help during development in terms of
style-guide compliance, but also to discover bugs (static code analysis).
```bash
  ./x/ci/run_linting.sh
```
Make sure to add any new analysis-relevant source file to the array declared in
the above Bash script.

In order to prevent erroneous merging, the CI-pipelines have to be manually triggered
before or upon MR issuing.

### Code Coverage
While it does not give a guarantee of absence of bugs, code coverage in
combination with unit-testing can be a powerful tool to discover vulnerabilities
in the code. A way to proceed is the following:
1. Diagnose the code coverage running `./x/local/diagnose_coverage.sh` from root directory;
2. Navigate to `src/test/htmlcov`;
3. Open `index.html` in your favorite browser;
4. Inspect coverages and implement new test scenarios for the missed lines of code.

Note: Some files will always have low coverage due to their nature, for instance
modules which mostly contain wrapper functions are ok to have low coverage.

Note: Coverage of 100% does not mean that everything is actually covered! For
instance, if you function does not check the arguments and therefore there is no
test checking the check-of-args the coverage will not suffer from this, despite
it being an uncovered part of SW.

### Code Optimization
In order to be able to optimize the code from a performance perspective, the
code (analysis) can be profiled and the profile visualized using `cProfile` and
`snakeviz`. Since this can be something which is done in an iterative way, a
bash script running the analysis on a local test dataset is available in `x/local/`,
namely `profile_analysis.sh`. Run it from root to inspect the analysis code
performance.

### Documentation
Document your code. To be aligned with the certification, code has to be
documented and we do this using `doxygen`. Check the `CONTRIBUTING` guidelines
and the [doxygen documentation](https://www.doxygen.nl/manual/docblocks.html#pythonblocks).
**We stick to Python native doc-strings with leading exclamation mark**
`"""!docstring"""` **for Doxygen formatting**.


## Doxygen Links
Here are the links to get to the main sources in the project (only works in
Doxygen view of the documentation):
* `run_remote_analysis` Main file running on VM;
* `MaurenbrecherStepDetection` Data segmentation and stride classification tools;
* `MaurenbrecherStepAnalysis` Strides in ROWs analysis;


## Folder Structure
This project features the following main-subfolders:
* [config/](config/) - Contains general configuration files (not analysis-specific configurations!) such as logging configurations and cronjob configurations.
* [debug/](debug/) - Destination folder for debugging outputs such as debug files and plots.
* [doc/](doc/) - Contains the (Doxygen) documentation. To see the documentation, generate it with `./x/local/generate_docu.sh` and then open `doc/html/index.html` in a browser.
* [exp/](exp/) - Contains experimental features which are being explored for improving Nushu.
  * [alternative_integration_strategies/](exp/alternative_integration_strategies/) Alternative integration strategies investigation.
  * [double_support_time/](exp/double_support_time/) Development of DST estimation.
  * [motion_detection/](exp/motion_detection/) Regions of motion detection development for automatic
  switching on and off of logging.
  * [sga_turning/](exp/sga_turning/) Seamless gait analysis (SGA) turning detection and analysis development.
  * [shared/](exp/shared/) Shared resources for experimental features, e.g. data loading and constants.
  * [stride_duration/](exp/stride_duration/) Stride duration reimplementation (based on TO-TO distance).
  * [walking_stability/](exp/walking_stability/) Walking stability evaluation investigation.
* [img/](img/) - Images used in markdown files and documentation.
* [labels/](labels/) - MD Certification labels.
* [logs/](logs/) - Default analysis logs destination folder.
* [nushu_pryv_pyclient/](nushu_pryv_pyclient/) - (Git Submodule) Client for communicating with the DB.
* [reports/](reports/) - Temporary folder for PDF reports and ZIP archives; needed for debug purposes.
* [src/](src/) - The main source folder: the main analysis tool source code is herein.
* [studies/](studies) - Collection of studies performed to validate Magnes Nushu, divided by premise/partner:
  * [cereneo/](cereneo/) Contains work related to validation of the system and experiments performed at Cereneo.
  * [downunder/](downunder/) Work related to JTI, project details are still TBD: this folder is ignored in the certified usage of Magnes Nushu.
  * [felix-platter/](felix-platter/) Contains the work related to the studies performed in collaboration with Felix-Platter Hospital in Basel.
  * [gaitrite-validation/](gaitrite-validation/) Contains the pipeline for validation of Nushu v. Gaitrite from April 2022.
  * [hirslanden/](hirslanden/) Contains investigations that were conducted on data collected by patients at the Hirslanden clinic in ZH.
  * [internal/](felix-internal/) Contains studies that were conducted on internally gathered datasets.
* [utils/](utils/) - Utility folder: a collection of tools useful for development and debugging:
  * [profiling/](utils/profiling/) Gait-analysis code profiling tool.
  * [sensor_calibration/](sensor_calibration/) - Contains work related to sensor calibration.
* [x/](x/) - Executables: mostly simple bash scripts for running stuff. They are internally divided in
  * [ci/](x/ci/) Scripts run during CI piplines (test execution) - can be tun locally.
  * [local/](x/local/) Scripts to run during development - only run locally.
  * [remote/](x/remote/) Scripts running once deployment is complete - only run on remote Debian machines.


## Relevant Source Files Call Tree
![whitebox-view](img/nushu/REM-whitebox-01.jpg)


## Gait Analysis Implementation

### Pipeline Description
The full pipeline of the automated analysis is the following:
0. The DB is inspected for data to be analyzed. For each dataset:
  1. Data for both, L&R, are fetched from the remote DB.
  2. For each side, candidate strides are sought. Candidate strides are called regions of interest (ROI).
  3. For each side, regions of walking (ROW) are sought. A ROW is defined to be an extended region of the data where significant activity is recognized.
  4. ROIs are tested against ROWs: ROIs which are not in a ROW are discarded.
  5. Each ROI, which passes this first test is fed through a classifier, which validates if the ROI is to be considered a STEP (stride) or not. Valid STEPs (strides) are collected into ROWs.
  6. Each STEP (stride) within a valid ROW is analyzed and results are collected grouped by ROW.
  7. A report based on the results is generated (given that at least one ROW has been detected).
  8. Results of all ROWs and the report are pushed to the DB.
1. For all users with at least one successfully analyzed session, the user statistics are recomputed.

_NOTE_ There is some mixing up of terminology when it comes to step vs. stride.
This is due an initial lack of knowledge in the technical terminology where the
developers thought them to be synonyms when in fact they are not. A stride is
the motion of a single leg, from rest to rest (so what we identify as a `Step`
object). A step (medically speaking) is the difference in position between L&R
feet once both are at rest. So, roughly speaking, the step length is about half
the stride length. There are plans to make the source code more consistent at
some point, but since this is just a matter of terminology and it does not
directly affect the analysis as long as it is known how to interpret the results
priority on this task is low - AS.

### Data Structures Overview

![raw-data](img/nushu/data-views_raw-data.png)

Raw data overview.

![results-data](img/nushu/data-views_results-data-01.jpg)

Results data overview.


## Contributors
In chronological order
- Kiran Kuruvithadam: GUI setup, analysis implementation.
- Jiaen Wu: BLE interface, live plotting, data-acquisition, analysis implementation, Cereneo studies, Online gait-phase detection algorithms (SVM).
- Alessandro Schaer: source code management, robustness, testing, review, CI, `Steps.py` module, documentation, `gait_analyis/`, configuration options, Cereneo studies, Felix-Platter studies, Downunder studies, validation studies, profiling, code coverage.
- Henrik Maurenbrecher: automated analysis implementation, sensor calibration, Cereneo studies, Felix-Platter studies, Hirslanden studies.
- Barna Becsek: BLE interface, testing, data analysis, web-interface.
- Francesco Buffa: web-interface (moved to NUSHU_native_web).
- Viviane Gerber: BLE interface improvements, sync-by-ROW enhancements, automatic NN retraining based on missed steps, improved NN performance.

![logo](img/logo.png)
