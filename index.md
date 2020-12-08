# Advent of Code 2020



## Day 1

<details>
<summary>
View
</summary>


</details>


## Day 2

<details>
<summary>
View
</summary>



### Part 1

<details>
<summary>
Code
</summary>

```python
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
```

</details>


### Part 2

<details>
<summary>
Code
</summary>

```python
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
```

</details>

</details>


## Day 3

<details>
<summary>
View
</summary>



### Part 1

<details>
<summary>
Code
</summary>

```python
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

```

</details>


### Part 2

<details>
<summary>
Code
</summary>

```python
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

```

</details>

</details>


## Day 4

<details>
<summary>
View
</summary>



### Part 1

<details>
<summary>
Code
</summary>

```python
import logging
import pathlib
import re
from typing import Dict, List, Optional

import click
import pydantic
from rich import print
from rich.logging import RichHandler


class Passport(pydantic.BaseModel):
    byr: str  # (Birth Year)
    iyr: int  # (Issue Year)
    eyr: int  # (Expiration Year)
    hgt: str  # (Height)
    hcl: str  # (Hair Color)
    ecl: str  # (Eye Color)
    pid: str  # (Passport ID)
    cid: Optional[str] = None  # (Country ID)


def solve(input_passports: List[Dict]) -> int:
    count_valid_passports = 0
    for passport in input_passports:
        try:
            Passport.parse_obj(passport)
            count_valid_passports += 1
        except Exception as e:
            logging.debug(e)

    return count_valid_passports


def parse_input(
    input_lines: List[str],
) -> List[Dict]:
    PASSPORT_PATTERN = r"(\w+):(#?\w+)"
    passports: List[Dict] = []
    passport: Dict = {}
    for line in input_lines:
        matches = re.findall(PASSPORT_PATTERN, line)
        if not matches:
            passports.append(passport)
            passport = {}
            continue

        passport_entries = {match[0]: match[1] for match in matches}
        passport.update(passport_entries)
    passports.append(passport)  # add last entry
    return passports


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
    print(f"[bold green]{solution}[/bold green] of the passports are valid.")


if __name__ == "__main__":
    cli()

```

</details>


### Part 2

<details>
<summary>
Code
</summary>

```python
import logging
import pathlib
import re
from typing import Dict, List, Optional

import click
import pydantic
from rich import print
from rich.logging import RichHandler

HEIGHT_REGEX = r"^(\d+)(cm|in)$"
HEIGHT_BOUNDS = {"cm": (150, 193), "in": (59, 76)}
HAIR_REGEX = r"^#[0-9a-f]{6}$"
EYE_COLORS = ("amb", "blu", "brn", "gry", "grn", "hzl", "oth")
PID_REGEX = r"^\d{9}$"


class Passport(pydantic.BaseModel):
    byr: int = pydantic.Field(ge=1920, le=2002)  # (Birth Year)
    iyr: int = pydantic.Field(ge=2010, le=2020)  # (Issue Year)
    eyr: int = pydantic.Field(ge=2020, le=2030)  # (Expiration Year)
    hgt: str = pydantic.Field(regex=HEIGHT_REGEX)  # (Height)
    hcl: str = pydantic.Field(regex=HAIR_REGEX)  # (Hair Color)
    ecl: str  # (Eye Color)
    pid: str = pydantic.Field(regex=PID_REGEX)  # (Passport ID)
    cid: Optional[str] = None  # (Country ID)

    @pydantic.validator("hgt")
    def check_height_values(cls, v):
        match = re.match(HEIGHT_REGEX, v)
        value, unit = match.group(1), match.group(2)
        if (int(value) >= HEIGHT_BOUNDS[unit][0]) and (
            int(value) <= HEIGHT_BOUNDS[unit][1]
        ):
            return v

        raise ValueError("Height {v} is invalid.")

    @pydantic.validator("ecl")
    def check_allowed_values(cls, v):
        if v in EYE_COLORS:
            return v
        raise ValueError("Eye color {v} is invalid.")


def solve(input_passports: List[Dict]) -> int:
    count_valid_passports = 0
    for passport in input_passports:
        print(passport)
        try:
            Passport.parse_obj(passport)
            count_valid_passports += 1
            print("Count increment +1")
        except Exception as e:
            logging.debug(e)

    return count_valid_passports


def parse_input(
    input_lines: List[str],
) -> List[Dict]:
    PASSPORT_PATTERN = r"(\w+):(#?\w+)"
    passports: List[Dict] = []
    passport: Dict = {}
    for line in input_lines:
        matches = re.findall(PASSPORT_PATTERN, line)
        if not matches:
            passports.append(passport)
            passport = {}
            continue

        passport_entries = {match[0]: match[1] for match in matches}
        passport.update(passport_entries)
    passports.append(passport)  # add last entry
    return passports


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
    print(f"[bold green]{solution}[/bold green] of the passports are valid.")


if __name__ == "__main__":
    cli()

```

</details>

</details>


## Day 5

<details>
<summary>
View
</summary>



### Part 1

<details>
<summary>
Code
</summary>

```python
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

```

</details>


### Part 2

<details>
<summary>
Code
</summary>

```python
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import List, Optional

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


def solve(boarding_passes: List[BoardingPass]) -> Optional[int]:
    seat_ids = {boarding_pass.id for boarding_pass in boarding_passes}
    for seat_id in range(min(seat_ids), max(seat_ids)):
        if seat_id not in seat_ids:
            return seat_id
    return None


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
    print(f"My seat ID is [bold green]{solution}[/bold green].")


if __name__ == "__main__":
    cli()

```

</details>

</details>
