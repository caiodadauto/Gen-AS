# Deep-Generative Graphs for the Internet (DGGI)
---

DGGI is a software initially designed to synthesize graphs that reproduce the structure observed in intra-AS networks.
Under the hood, DGGI uses the model [GraphRNN]() to generate the synthetic graphs. [MLflow]() is used to track the
quality of training process and to save the best models, and [Hydra]() is used to control the configuration parameters.
The pre-trained generator for intra-AS graphs synthesizing are provided [here](), these training procedure uses the
dataset provided [here](), which was introduced in the following paper:
```
title={Data-driven Intra-Autonomous Systems Graph Generator}
author={Caio Vinicius Dadauto and Nelson Luis Saldanha da Fonseca and Ricardo da Silva Torres}
year={2023}
eprint={2308.05254}
archivePrefix={arXiv}"
primaryClass={cs.NI}"
```

This software provides an easy way to:
1. train the model for any new graph dataset;
2. generate graphs based on a trained model;
3. evaluate different versions of generators;


## 1. Installation
DGGI installation relies on [poetry]() and [mamba](), both can be installed as follow:
```bash
curl -sSL https://install.python-poetry.org | python3 -
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```
for more details about this installation process, check [here]() and [here]().

Once `poetry` and `mamba` have been installed, in the root directory of this project, do
```bash
mamba env create -f environment.yaml
poetry install
```
This will install the required packages, add the default configuration file to `~/.config/dggi_dggm/config.yaml`,
and create two CLI commands `dggi` and `dggi-gui`.

## 2. Configuration
All runs of DGGI will consider the default configuration specified in `~/.config/dggi_dggm/config.yaml`, any modification
of this file will impact all runs of DGGI. To limit the scope of the configuration to a specific location, the default
configuration should be copied to this location, so, every time DGGI runs in this location, the copied configuration will be
considered, instead of default file in `~/.config/dggi_dggm`.

The configuration parameters are detailed in the table below.

### 2.1. For Hydra
| Parameter Name | Description                      | Default Value |
| -------------- | -------------------------------- | ------------- |
| `run.dir`        | Hydra default run directory      | "."           |
| `output_subdir`  | Location to persist Hydra output | null          | 

### 2.2. For MLflow
| Parameter Name | Description                    | Default Value |
| -------------- | ------------------------------ | ------------- |
| `exp_name`       | MLflow experiment name         | "DGGI         |
| `exp_tags`       | MLflow tags for the experiment | null          |
| `run_tags`       | MLflow tags for the run        | null          |
| `run_id`         | MLflow run ID                  | null          |

### 2.3. For Input Data
| Parameter Name | Description                                                                                                                                | Default Value |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| `num_workers`    | Number of threads used to load the data                                                                                                    | `4`           |
| `data_size`      | Total number of samples in the dataset                                                                                                     | `90326`       |
| `batch_size`     | The size of the batches used during the training                                                                                           | `40`          |
| `min_num_node`   | The number of nodes of the graph with smallest number of nodes in the dataset                                                              | `12`          |
| `max_num_node`   | The number of nodes of the graph with largest number of nodes in the dataset                                                               | `250`         |
| `max_prev_node`  | The maximum number of edge predictions that should be made for each incoming node. If `null`, DGGI will estimate this value automatically. | `250`         |
| `check_size`     | Check if the loaded graphs have the number of nodes between `min_num_node` and `max_num_node`                                              | `false`       |
| `inplace`        | If `true`, the graphs will not be loaded to the memory, otherwise, all graphs will be in-memory.                                           | `false`       |
| `source_path`    | The location of dataset.                                                                                                                   | `./data`      |


### 2.4. For the Model (GraphRNN)
| Parameter Name            | Description                                                                                      | Default Value |
| ------------------------- | ------------------------------------------------------------------------------------------------ | ------------- |
| `hidden_size_rnn`           | Hidden size for GRU for graph embedding. For small versions, 64 is recommended.                  | `128`           |
| `hidden_size_rnn_output`    | Hidden size for GRU for prediction connections.                                                  | `16`            |
| `embedding_size_rnn`        | The dimension of latent space of GRU for graph embedding. For small versions, 32 is recommended. | `64`            |
| `embedding_size_rnn_output` | The dimension of latent space of GRU for prediction connections.                                 | `8`             |
| `embedding_size_output`     | The dimension of latent space of MLP for the output. For small versions, 32 is recommended.      | `64`            |
| `num_layer`                 | The number of GRU layers.                                                                        | `4`             |


### 2.5. For Training
| Parameter Name      | Description                                                                                                                                               | Default Value |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `seed`                | The seed for random generator initial state.                                                                                                              | `12345`         |
| `num_epochs`          | Number of epochs.                                                                                                                                         | `1000`          |
| `epochs_test`         | The model will be evaluate for every `epochs_test` epochs.                                                                                                | `100`           |
| `epochs_test_start`   | Number of epochs to start considering the evaluation.                                                                                                     | `1`             |
| `test_batch_size`     | How many graphs will be generate per model iteration.                                                                                                     | `40`            |
| `test_total_size`     | How many graphs will be used for evaluation.                                                                                                              | `500`           |
| `lr`                  | Initial learning rate.                                                                                                                                    | `0.003`         |
| `lr_rate`             | The rating that the learning rate will be decreased for each milestone.                                                                                   | `0.3`           |
| `n_checkpoints`       | The number of model checkpoints to be persisted.                                                                                                          | `4`             |
| `milestones`          | A list of number of epochs used as milestones to control the learning rate decreasing.                                                                    | `[300, 500]`    |
| `n_bootstrap_samples` | The number of the test graphs should be sampled from test set.                                                                                            | `100`           |
| `metrics`             | A list of metrics that should be used during the evaluation. Possible metrics are: `degree`, `clustering`, `assortativity`, `betweenness`, and `pagerank` | `["degree"]`    |


### 2.6. For Generation
| Parameter Name  | Description                                                                                                                | Default Value     |
| --------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `seed`            | The seed for random generator initial state.                                                                               | `12345`             |
| `test_batch_size` | How many graphs will be generate per model iteration.                                                                      | `40`                |
| `test_total_size` | How many graphs will be generated.                                                                                         | `500`               |
| `min_num_node`    | The minimum number of nodes required for the graphs to be generated. If `null`, uses the value defined for the input data. | `null`            |
| `max_num_node`    | The maximum number of nodes required for the graphs to be generated. If `null`, uses the value defined for the input data. | `null`            | 
| `save_dir`        | The location where the synthetic graphs will be saved.                                                                     | 'dggi_generation' |


### 2.7. For Evaluation
| Parameter Name  | Description                                                                                                                                               | Default Value                                                        |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `seed`            | The seed for random generator initial state.                                                                                                              | `12345`                                                                |
| `test_batch_size` | How many graphs will be generate per model iteration.                                                                                                     | `40`                                                                   |
| `test_total_size` | How many graphs will be used for evaluation.                                                                                                              | `500`                                                                  |
| `metrics`         | A list of metrics that should be used during the evaluation. Possible metrics are: `degree`, `clustering`, `assortativity`, `betweenness`, and `pagerank` | `["degree", "clustering", "assortativity", "betweenness", "pagerank"]` |


## 2. Usage
DGGI can be used either via command line or via user interface.

### 2.1. Command Line
#### 2.1.1. For Training
To train a model from scratch, do
```bash
dggi train
```

#### 2.1.2. For Generation
To generate synthetic graphs using a trained model, do
```bash
dggi generate
```

#### 2.1.3. For Evaluation
To evaluate a previous trained model, do
```bash
dggi evaluate
```


### 2.1. User Interface
On terminal, do
```bash
dggi-gui
```

#### 2.1.1. For Training
To train a model from scratch, do
```bash
dggi train
```

#### 2.1.2. For Generation
To generate synthetic graphs using a trained model, do
```bash
dggi generate
```

#### 2.1.3. For Evaluation
To evaluate a previous trained model, do
```bash
dggi evaluate
```
