# Deep-Generative Graphs for the Internet (DGGI)
---

DGGI is a software initially designed to synthesize graphs that reproduce the structure observed in intra-AS networks.
Under the hood, DGGI uses the model GraphRNN to generate the synthetic models.
The pre-trained weights for intra-AS graphs generation are provided [here](), these weights were obtained through a training
procedure using the dataset provided [here]().

DGGI delivery an ease way to:
1. train the model for a new dataset;
2. generate graphs based on a trained model;
3. evaluate different versions of a trained model; 

## 1. Installation
DGGI installation relies on [poetry]() and [mamba](), both can be installed as follow:
```bash
curl -sSL https://install.python-poetry.org | python3 -
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

Once `poetry` and `mamba` have been installed, in the root directory of this project, do
```bash
mamba env create -f environment.yaml
poetry install
```
This will install the package, add the default configuration file to `~/.config/dggi_dggm/config.yaml`, and create two CLI commands `dggi` and `dggi-gui`.

## 2. Configuration
In order to define local configurations, the default configuration file just need to be copied
to the work directory. Now, all runs of the `dggi` command will consider the parameters in the configuration file
located in the current directory.

## 2. Usage
### 2.1 Training
To train a model from scratch, do
```bash
dggi train
```

### 2.1 Generation
To generate synthetic graphs using a trained model, do
```bash
dggi generate
```

### 2.1 Generation
To evaluate a previous trained model, do
```bash
dggi evaluate
```
