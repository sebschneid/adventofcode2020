import itertools
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, NewType, Optional, Tuple

import pydantic
from rich import print


class PasswordPolicy(pydantic.BaseModel):
    occurence_min: int
    occurence_max: int
    character: str = pydantic.Field(max_length=1)


Password = NewType("Password", str)


def solve(puzzle_input: List[Tuple[PasswordPolicy, Password]]) -> int:
    counter_valid = 0
    for policy, password in puzzle_input:
        occurences_in_password = len(re.findall(policy.character, password))
        if (
            occurences_in_password >= policy.occurence_min
            and occurences_in_password <= policy.occurence_max
        ):
            counter_valid += 1
        else:
            print(
                f"[bold red]Invalid[/bold red] password: [bold blue]{password}[/bold blue]\n"
                f"[bold red]{occurences_in_password}[/bold red] is outside of "
                f"[{policy.occurence_min},{policy.occurence_max}]\n"
            )
    return counter_valid


def parse_line(line: str) -> Optional[Tuple[PasswordPolicy, Password]]:
    REGEX_PATTERN = r"(\d+)-(\d+) (\D): (\D+)\n"
    match = re.match(REGEX_PATTERN, line)
    if match is None:
        return None
    policy = PasswordPolicy(
        occurence_min=match.group(1),
        occurence_max=match.group(2),
        character=match.group(3),
    )
    password = Password(match.group(4))
    return (policy, password)


def parse_input(
    input_lines: List[str],
) -> List[Tuple[PasswordPolicy, Password]]:
    puzzle_input = []
    for line in input_lines:
        result = parse_line(line)
        if result is not None:
            puzzle_input.append(result)
    return puzzle_input


if __name__ == "__main__":
    input_file = pathlib.Path(".") / sys.argv[1]
    with open(input_file) as open_file:
        puzzle_input = parse_input(open_file.readlines())
    solution = solve(puzzle_input)
    print(f"The solution to this puzzle is [bold green]{solution}")