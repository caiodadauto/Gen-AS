# Deep Internet Graph Generator (DIGG)
---

## 1. Installation
DIGG installation relies on [poetry]() and [mamba](), both can be installed as follow:
```bash
curl -sSL https://install.python-poetry.org | python3 -
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

Once `poetry` and `mamba` have been installed, in the root directory of this project, do
```bash
mamba create -n test --file conda-linux-64.lock
poetry install
```
This will install the package, create the CLI command `digg`, and add the default configuration file to `~/.config/digg_dggm/config.yaml`.

## 2. Configuration
In order to define local configurations, the default configuration file just need to be copied
to the work directory. Now, all runs of the `digg` command will consider the parameters in the configuration file
located in the current directory.

## 2. Usage
### 2.1 Training
To train a model from scratch, do
```bash
digg train
```

### 2.1 Generation
To generate synthetic graphs using a trained model, do
```bash
digg generate
```

### 2.1 Generation
To evaluate a previous tranined model, do
```bash
digg evaluate
```
