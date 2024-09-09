# Master Thesis Project

## Overview

This repository contains code for the master thesis:

_Efficient Resource Allocation for Distributed Dataflows using Contextual Performance Modeling_.


## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Installation](#local_installation)
    1. [Makefile](#makefile)
    2. [Installation Workflow](#installation_workflow)
3. [Run](#run)
4. [References](#reference)
5. [Contact](#contact)

## <a id="prerequisites"></a>1. Prerequisites

- `make` utility installed on your system. 

## <a id="local_installation"></a>2. Local Installation

The provided [Makefile](Makefile) will help you setting up the project and necessary dependencies locally. 

### <a id="makefile"></a>2.1 Makefile

| Target             | Description                                                                            |
|--------------------|----------------------------------------------------------------------------------------|
| `make`             | `make help` is executed.                                                               |
| `make all`         | Dummy target.                                                                          |
| `make clean`       | Deletes the Python virtual environment.                                                |
| `make format`      | Formats code under the [`model/src/`](/model/src/) directory.                                         |
| `make poetry`      | Installs Python dependencies using poetry.                                             |
| `make init_poetry` | Initializes poetry for the existing project.                                           |
| `make add_package` | Adds a new package using poetry. Usage: `make add_package PACKAGE=[NAME_OF_PACKAGE]`.  |
| `make venv`        | Creates a Python virtual environment and installs poetry.                              |


### <a id="installation_workflow"></a>2.2 Installation Workflow

In the root directory of the project execute

```
make venv
```

to install the virtual environment and poetry in it. 

Afterwards, activate the `venv` inside the [`model`](model/) directory:

```
source .venv/bin/activate
```


Then execute 

```
make poetry
```

inside the root directory of the project, to install all additional necessary dependencies.


## <a id="run"></a>3. Run

To run the experiments for the project locally, execute the `run_experiment.py` script under [`model/evaluation/`](/model/evaluation/).

The script accepts the following arguments:

| Argument             | Type   | Description                                  | Required |
|----------------------|--------|----------------------------------------------|----------|
| `--experiment-name`  | `str`  | Name of experiment.                          | Yes      |
| `--dataset`          | `str`  | Name of the underlying dataset (e.g. c3o, scout)                         | Yes      |
| `--num-configs`      | `int`  | Number of configurations to investigate.     | No       |
| `--algorithms`       | `list` | Algorithms/workloads for the experiment run.                           | Yes      |
| `--num-samples`      | `int`  | Number of samples to draw.                   | No       |
| `--num-iters`        | `int`  | Number of iterations. Default is 200.        | No       |


## <a id="reference"></a>4. References

The thesis builds upon an existing runtime prediction approach called [_Bellamy_](https://ieeexplore.ieee.org/abstract/document/9555951). Therefore, some of the code in this repository originates from [https://github.com/dos-group/bellamy-runtime-prediction/tree/main](https://github.com/dos-group/bellamy-runtime-prediction/tree/main)

## <a id="contact"></a>5. Contact

If there are any problems related to installation or configuration of the project, please reach out to [me](https://kronsbein.github.io).