import itertools
import math
import pathlib
import sys
from typing import List

from rich import print


def solve(puzzle_input: List[int]):
    combinations = itertools.combinations(puzzle_input, 3)
    combinations_solution = [
        combination for combination in combinations if sum(combination) == 2020
    ][0]
    solution = math.prod(combinations_solution)
    return solution


def read_input(input_file: pathlib.Path) -> List[int]:
    with open(input_file) as open_file:
        puzzle_input = [int(line) for line in open_file]
    return puzzle_input


if __name__ == "__main__":
    input_file = pathlib.Path(".") / sys.argv[1]
    puzzle_input = read_input(input_file)
    solution = solve(puzzle_input)
    print(f"The solution to this puzzle is [bold green]{solution}")