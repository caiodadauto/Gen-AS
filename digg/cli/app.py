from os import getcwd, listdir
from os.path import join, expanduser

import click
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from digg.generator.training import train as _train
from digg.generator.evaluation import generate as _generate


@click.group()
@click.pass_context
@click.option(
    "-cd",
    "--config-dir",
    type=str,
    default=None,
    help="Path to the directory with configuraion file. If not provided, the current directory will be considered if a file `CONFIG_NAME.yaml` exists, otherwise, `$HOME/.config/digg_dggm/` will be considered.",
)
@click.option(
    "-cn",
    "--config-name",
    type=str,
    default=None,
    help="Configuration file name without the extension `yaml`. If not provided, `config` will be used.",
)
@click.option(
    "-mlfd",
    "--mlf-dir",
    type=str,
    default=None,
    help="Path to the directory to be used to initialize the MLFlow tracking. If not provided, the current directory will be considered."
)
def main(ctx, config_dir, config_name, mlf_dir):
    ctx.ensure_object(dict)
    cwd = getcwd()
    mlf_dir = cwd if mlf_dir is None else mlf_dir
    config_name = "config" if config_name is None else config_name
    if config_dir is None:
        if config_name + ".yaml" in [p for p in listdir(cwd) if p.endswith("yaml")]:
            config_dir = cwd
        else:
            config_dir = join(expanduser("~"), ".config", "digg_dggm")
    ctx.obj["cwd"] = cwd
    ctx.obj["config_dir"] = config_dir
    ctx.obj["config_name"] = config_name
    ctx.obj["mlf_dir"] = mlf_dir


@main.command()
@click.pass_context
def train(ctx):
    """Train the model based on a digg configuration."""
    initialize_config_dir(
        version_base=None, config_dir=ctx.obj["config_dir"], job_name="train"
    )
    cfg = compose(config_name=ctx.obj["config_name"])
    _train(cfg, ctx.obj["mlf_dir"])

@main.command()
@click.pass_context
@click.option(
    "-m",
    "--min-number-nodes",
    type=int,
    default=None,
    help="Minimum number of nodes. If not provided, the value in configuration file will be considered.",
)
@click.option(
    "-M",
    "--max-number-nodes",
    type=int,
    default=None,
    help="Maximum number of nodes. If not provided, the value in configuration file will be considered.",
)
@click.option(
    "-n",
    "--number-graphs",
    type=int,
    default=None,
    help="Number of synthetic graphs to be generated. If not provided, the value in configuration file will be considered."
)
def generate(ctx, min_number_nodes, max_number_nodes, number_graphs):
    """Generate synthetic graphs based on the digg configurations."""
    initialize_config_dir(
        version_base=None, config_dir=ctx.obj["config_dir"], job_name="train"
    )
    cfg = compose(config_name=ctx.obj["config_name"])
    _generate(cfg, number_graphs, min_number_nodes, max_number_nodes, ctx.obj["mlf_dir"])

# @click.command()
# @click.option(
#     "-c",
#     "--config-path",
#     type=str,
#     default=join(expanduser("~"), ".config", "digg_dggm"),
# )
# @click.option("-n", "--config-name", type=str, default="config")
# def evaluate(config_dir, config_name):
#     """Evaluate a trained model based on the digg configurations.
#
#     \b
#     Args:
#         CONFIG_DIR: Path to the directory with configuraion file, default `$HOME/.config/digg_dggm/`
#         CONFIG_NAME: Configuration file name, default `config.yaml`
#     """
#
#     with initialize_config_dir(
#         config_dir=config_dir, version_base=None, job_name="evaluate"
#     ):
#         cfg = compose(config_name=config_name)
#         click.echo(f"Evaluation")
#
#

if __name__ == "__main__":
    main()
