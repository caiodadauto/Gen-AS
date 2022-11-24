import click


@click.group()
def app():
    pass


@click.command()
@click.argument("training-data-path", type=str)
def training(training_data_path):
    """This a command to train the model.

    \b
    Args:
        TRAINING_DATA_PATH: The path to the trainig data.
    """
    click.echo(f"{training_data_path}")


@click.command()
@click.argument("metric", type=str)
def evaluation(metric):
    click.echo(f"{metric}")


app.add_command(training)
app.add_command(evaluation)


if __name__ == "__main__":
    app()
