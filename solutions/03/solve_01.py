import math
import pathlib
import sys
from dataclasses import dataclass
from typing import List, Set

import numpy as np  # type: ignore
from rich import print


@dataclass(frozen=True)
class Position:
    x: int
    y: int


@dataclass
class Field:
    width: int
    height: int
    trees: Set[Position]


def solve(field: Field, step_x=3, step_y=1) -> int:
    # get positions traveled
    steps_to_bottom = field.height // step_y
    positions_traveled = {
        Position(x=x, y=y)
        for x, y in zip(
            np.arange(0, steps_to_bottom * step_x, step_x),
            np.arange(0, steps_to_bottom * step_y, step_y),
        )
    }

    # get all positions of relevant trees
    width_traveled = steps_to_bottom * step_x
    fields_needed = math.ceil(width_traveled / field.width)
    trees_full = {
        Position(x=tree.x + n_field * field.width, y=tree.y)
        for tree in field.trees
        for n_field in range(0, fields_needed)
    }

    # intersection
    count_trees_passed = len(positions_traveled.intersection(trees_full))
    return count_trees_passed


def parse_input(
    input_lines: List[str],
) -> Field:
    trees = {
        Position(x=x, y=y)
        for y, line in enumerate(input_lines)
        for x, char in enumerate(line)
        if char == "#"
    }
    height = len(input_lines)
    width = len(input_lines[0])
    return Field(width=width, height=height, trees=trees)


if __name__ == "__main__":
    input_file = pathlib.Path(".") / sys.argv[1]
    with open(input_file) as open_file:
        puzzle_input = parse_input(open_file.read().splitlines())
    solution = solve(puzzle_input)
    print(
        f"I encounter [bold green]{solution}[/bold green] trees "
        "following a slope of right 3 and down 1."
    )
