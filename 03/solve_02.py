import math
import pathlib
import sys
from dataclasses import dataclass
from typing import List, Set, Tuple

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


def count_trees_for_slope(field: Field, step_x=3, step_y=1) -> int:
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


def solve(
    field: Field,
    slopes: List[Tuple[int, int]] = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)],
):
    solution = 1
    for slope in slopes:
        count_trees = count_trees_for_slope(
            field, step_x=slope[0], step_y=slope[1]
        )
        solution *= count_trees
        print(
            f"Found [bold green]{count_trees}[/bold green] trees for the slope"
            f" ({slope[0]} right, {slope[1]} down)."
        )
    return solution


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
        f"Multiplication of the number of trees encountered along the "
        f"different slopes, results in [bold green]{solution}[/bold green]."
    )
