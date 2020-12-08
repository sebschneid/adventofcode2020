import logging
import pathlib
import re
from dataclasses import dataclass
from typing import List

import click
from rich import print
from rich.logging import RichHandler

row_to_binary = {"F": "0", "B": "1"}
col_to_binary = {"L": "0", "R": "1"}


@dataclass
class BoardingPass:
    row: int
    col: int
    id: int

    @classmethod
    def create_from_code(cls, code: str):
        REGEX_SEAT = r"^([FB]{7})([LR]{3})$"
        match = re.match(REGEX_SEAT, code)
        if match is None:
            raise ValueError(f"Code {code} does not have the required format")

        row_binary = "".join(map(row_to_binary.get, match.group(1)))
        col_binary = "".join(map(col_to_binary.get, match.group(2)))
        row = int(row_binary, 2)
        col = int(col_binary, 2)
        id = row * 8 + col
        return cls(row=row, col=col, id=id)


def solve(boarding_passes: List[BoardingPass]) -> int:
    return max([boarding_pass.id for boarding_pass in boarding_passes])


def parse_input(
    input_lines: List[str],
) -> List[BoardingPass]:
    boarding_passes = [
        BoardingPass.create_from_code(line) for line in input_lines
    ]
    return boarding_passes


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
        "Highest seat ID on a boarding pass is "
        f"[bold green]{solution}[/bold green]."
    )


if __name__ == "__main__":
    cli()
