from pathlib import Path

training: Path = Path(__file__).absolute().parent.parent

project: Path = training.parent

data: Path = project/'data'
GAN_out: Path = data/'GAN_output'
datasets: Path = data/'datasets'
logs: Path = data/'logs'

for i in list(vars().values()):
    if isinstance(i, Path):
        i.mkdir(parents=True, exist_ok=True)
        