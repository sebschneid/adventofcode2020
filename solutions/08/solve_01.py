import logging
import pathlib
from typing import List

import click
from console import Instruction, Operator
from rich import print
from rich.logging import RichHandler


def solve(instructions: List[Instruction]) -> int:
    operator = Operator(instructions)
    operator.run()
    return operator.accumulator


def parse_input(
    input_lines: List[str],
) -> List[Instruction]:
    instructions = [Instruction.from_string(line) for line in input_lines]
    return instructions


@click.command()
@click.argument("input_filename", type=str)
@click.option("--debug", type=bool, default=False, is_flag=True)
def cli(input_filename, debug):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, handlers=[RichHandler()])
    input_file = pathlib.Path(".") / input_filename

    with open(input_file) as open_file:
        puzzle_input = parse_input(open_file.read().splitlines())
    solution = solve(puzzle_input)

    print(
        "Accumulator before the first operation is executed twice is "
        f"[bold green]{solution}[/bold green]."
    )


if __name__ == "__main__":
    cli()
