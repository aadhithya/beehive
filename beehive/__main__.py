from loguru import logger
from typer import Argument, Option, Typer

app = Typer(name="beehive")


@app.command("train", help="Train a model.")
def train():
    pass


@app.command("eval", help="Evaluate a trained model.")
def eval():
    pass


@app.command("infer", help="Run inference using a trained model.")
def infer():
    pass


if __name__ == "__main__":
    app()
