import logging
import pathlib
from typing import List, Optional

import click
from console import Instruction, Operation, Operator
from rich import print
from rich.logging import RichHandler


def solve(instructions: List[Instruction]) -> Optional[int]:
    for i, instruction in enumerate(instructions):
        if (operation := instruction.operation) in (
            Operation.jmp,
            Operation.nop,
        ):
            changed_operation = (
                Operation.jmp if operation == Operation.nop else Operation.nop
            )
            # change instruction
            instructions[i] = Instruction(
                operation=changed_operation, argument=instruction.argument
            )
            operator = Operator(instructions)
            instructions_valid = operator.validate_instructions()
            if instructions_valid:
                return operator.accumulator

            # change to original
            instructions[i] = instruction

    return None


def parse_input(
    input_lines: List[str],
):
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
        "The accumulator after the fixed program terminates is "
        f"[bold green]{solution}[/bold green]."
    )


if __name__ == "__main__":
    cli()
