import itertools
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, NewType, Optional, Tuple

import pydantic
from rich import print


class PasswordPolicy(pydantic.BaseModel):
    position_first: int
    position_second: int
    character: str = pydantic.Field(max_length=1)


Password = NewType("Password", str)


def solve(puzzle_input: List[Tuple[PasswordPolicy, Password]]) -> int:
    counter_valid = 0
    for policy, password in puzzle_input:
        char_in_first_position = (
            policy.character == password[policy.position_first - 1]
        )
        char_in_second_position = (
            policy.character == password[policy.position_second - 1]
        )
        if char_in_first_position != char_in_second_position:
            counter_valid += 1
        else:
            print(
                f"[bold red]Invalid[/bold red] password: [bold blue]{password}[/bold blue]\n"
                f"[bold red]{policy.character}[/bold red] searched in either position {policy.position_first} or {policy.position_second}\n"
                f"Position {policy.position_first}: [bold red]{password[policy.position_first - 1]}[/bold red]\n"
                f"Position {policy.position_second}: [bold red]{password[policy.position_second - 1]}[/bold red]\n"
            )
    return counter_valid


def parse_line(line: str) -> Optional[Tuple[PasswordPolicy, Password]]:
    REGEX_PATTERN = r"(\d+)-(\d+) (\D): (\D+)\n"
    match = re.match(REGEX_PATTERN, line)
    if match is None:
        return None
    policy = PasswordPolicy(
        position_first=match.group(1),
        position_second=match.group(2),
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