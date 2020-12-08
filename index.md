# Advent of Code 2020



## Day 1



### solve_01.py

<details><summary>Code</summary>

```python
import itertools
import pathlib
import sys
from typing import List

import numpy as np
from rich import print


def solve(puzzle_input: List[int]):
    combinations = itertools.combinations(puzzle_input, 2)
    combination_solution = [
        combination for combination in combinations if sum(combination) == 2020
    ][0]
    solution = combination_solution[0] * combination_solution[1]
    return solution


def read_input(input_file: pathlib.Path) -> List[int]:
    with open(input_file) as open_file:
        puzzle_input = [int(line) for line in open_file]
    return puzzle_input


def read_input_numpy(input_file: pathlib.Path) -> np.array:
    puzzle_input = np.loadtxt(input_file, dtype=int)
    return puzzle_input


if __name__ == "__main__":
    input_file = pathlib.Path(".") / sys.argv[1]
    puzzle_input = read_input_numpy(input_file)
    solution = solve(puzzle_input)
    print(f"The solution to this puzzle is [bold green]{solution}")
```

</details>
</br>



### solve_02.py

<details><summary>Code</summary>

```python
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
```

</details>
</br>



</details>

</br>
<hr>
</br>


## Day 2



### solve_01.py

<details><summary>Code</summary>

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
</br>



### solve_02.py

<details><summary>Code</summary>

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
</br>



### test_01.py

<details><summary>Code</summary>

```python
from solve_01 import parse_input, solve

TEST_INPUT = ["1-3 a: abcde\n", "1-3 b: cdefg\n", "2-9 c: ccccccccc\n"]


def test_solution():
    result = solve(parse_input(TEST_INPUT))
    assert 2 == result

```

</details>
</br>



### test_02.py

<details><summary>Code</summary>

```python
from solve_02 import parse_input, solve

TEST_INPUT = ["1-3 a: abcde\n", "1-3 b: cdefg\n", "2-9 c: ccccccccc\n"]


def test_solution():
    result = solve(parse_input(TEST_INPUT))
    assert 1 == result

```

</details>
</br>



</details>

</br>
<hr>
</br>


## Day 3



### solve_01.py

<details><summary>Code</summary>

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
</br>



### solve_02.py

<details><summary>Code</summary>

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
</br>



### test_01.py

<details><summary>Code</summary>

```python
from solve_01 import parse_input, solve

TEST_INPUT = (
    """
    ..##.......
    #...#...#..
    .#....#..#.
    ..#.#...#.#
    .#...##..#.
    ..#.##.....
    .#.#.#....#
    .#........#
    #.##...#...
    #...##....#
    .#..#...#.#
    """
).split()


def test_solution():
    print(TEST_INPUT)
    result = solve(parse_input(TEST_INPUT))
    assert 7 == result

```

</details>
</br>



### test_02.py

<details><summary>Code</summary>

```python
from solve_02 import parse_input, solve

TEST_INPUT = (
    """
    ..##.......
    #...#...#..
    .#....#..#.
    ..#.#...#.#
    .#...##..#.
    ..#.##.....
    .#.#.#....#
    .#........#
    #.##...#...
    #...##....#
    .#..#...#.#
    """
).split()


def test_solution():
    print(TEST_INPUT)
    result = solve(parse_input(TEST_INPUT))
    assert 336 == result

```

</details>
</br>



</details>

</br>
<hr>
</br>


## Day 4



### solve_01.py

<details><summary>Code</summary>

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
</br>



### solve_02.py

<details><summary>Code</summary>

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
</br>



### test_01.py

<details><summary>Code</summary>

```python
from solve_01 import parse_input, solve

TEST_INPUT = (
    """
    ecl:gry pid:860033327 eyr:2020 hcl:#fffffd
    byr:1937 iyr:2017 cid:147 hgt:183cm

    iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884
    hcl:#cfa07d byr:1929

    hcl:#ae17e1 iyr:2013
    eyr:2024
    ecl:brn pid:760753108 byr:1931
    hgt:179cm

    hcl:#cfa07d eyr:2025 pid:166559648
    iyr:2011 ecl:brn hgt:59in
    """
).split("\n")


def test_solution():
    result = solve(parse_input(TEST_INPUT))
    assert 2 == result

```

</details>
</br>



### test_02.py

<details><summary>Code</summary>

```python
from solve_02 import parse_input, solve

TEST_INPUT = (
    """
    eyr:1972 cid:100
    hcl:#18171d ecl:amb hgt:170 pid:186cm iyr:2018 byr:1926

    iyr:2019
    hcl:#602927 eyr:1967 hgt:170cm
    ecl:grn pid:012533040 byr:1946

    hcl:dab227 iyr:2012
    ecl:brn hgt:182cm pid:021572410 eyr:2020 byr:1992 cid:277

    hgt:59cm ecl:zzz
    eyr:2038 hcl:74454a iyr:2023
    pid:3556412378 byr:2007

    pid:087499704 hgt:74in ecl:grn iyr:2012 eyr:2030 byr:1980
    hcl:#623a2f

    eyr:2029 ecl:blu cid:129 byr:1989
    iyr:2014 pid:896056539 hcl:#a97842 hgt:165cm

    hcl:#888785
    hgt:164cm byr:2001 iyr:2015 cid:88
    pid:545766238 ecl:hzl
    eyr:2022

    iyr:2010 hgt:158cm hcl:#b6652a ecl:blu byr:1944 eyr:2021 pid:093154719
    """
).split("\n")


def test_solution():
    result = solve(parse_input(TEST_INPUT))
    assert 4 == result

```

</details>
</br>



</details>

</br>
<hr>
</br>


## Day 5



### solve_01.py

<details><summary>Code</summary>

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
</br>



### solve_02.py

<details><summary>Code</summary>

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
</br>



### test_01.py

<details><summary>Code</summary>

```python
from solve_01 import parse_input, solve

TEST_INPUT = (
    """
    BFFFBBFRRR
    FFFBBBFRRR
    BBFFBBFRLL
    """
).split()


def test_solution():
    boarding_passes = parse_input(TEST_INPUT)
    assert boarding_passes[0].row == 70
    assert boarding_passes[0].col == 7
    assert boarding_passes[0].id == 567

    assert boarding_passes[1].row == 14
    assert boarding_passes[1].col == 7
    assert boarding_passes[1].id == 119

    assert boarding_passes[2].row == 102
    assert boarding_passes[2].col == 4
    assert boarding_passes[2].id == 820

    result = solve(parse_input(TEST_INPUT))
    assert 820 == result

```

</details>
</br>



</details>

</br>
<hr>
</br>


## Day 6



### solve_01.py

<details><summary>Code</summary>

```python
import pathlib
from typing import List, NewType, Set

import click
from rich import print

AnswersOfAnyoneInGroups = NewType("AnswersOfAnyoneInGroups", Set[str])


def solve(answers_of_groups: List[AnswersOfAnyoneInGroups]) -> int:
    return sum([len(answers) for answers in answers_of_groups])


def parse_input(
    input_lines: List[str],
) -> List[AnswersOfAnyoneInGroups]:
    answers_of_groups: List[AnswersOfAnyoneInGroups] = []
    answers: Set = set({})
    for line in input_lines:
        answers = answers.union(set(line))
        if len(line.strip()) == 0:
            answers_of_groups.append(AnswersOfAnyoneInGroups(answers))
            answers = set({})
            continue
    answers_of_groups.append(AnswersOfAnyoneInGroups(answers))
    return answers_of_groups


@click.command()
@click.argument("input_filename", type=str)
@click.option("--debug", type=bool, default=False, is_flag=True)
def cli(input_filename, debug):
    input_file = pathlib.Path(".") / input_filename
    with open(input_file) as open_file:
        puzzle_input = parse_input(open_file.read().splitlines())
    solution = solve(puzzle_input)
    print(
        f"The sum of the counts per group of questions to which "
        "[bold red]anyone[/bold red] "
        f"answered 'yes' is [bold green]{solution}[/bold green]."
    )


if __name__ == "__main__":
    cli()

```

</details>
</br>



### solve_02.py

<details><summary>Code</summary>

```python
import pathlib
from typing import List, NewType, Set

import click
from rich import print

AnswersOfAllInGroup = NewType("AnswersOfAllInGroup", Set[str])


def solve(answers_of_groups: List[AnswersOfAllInGroup]) -> int:
    return sum([len(answers) for answers in answers_of_groups])


def parse_input(
    input_lines: List[str],
) -> List[AnswersOfAllInGroup]:
    answers_of_groups: List[AnswersOfAllInGroup] = []
    start_new_group: bool = True
    for line in input_lines:
        if start_new_group:
            answers = set(line)
            start_new_group = False
        elif len(line.strip()) == 0:
            answers_of_groups.append(AnswersOfAllInGroup(answers))
            start_new_group = True
        else:
            answers = answers.intersection(set(line))
    answers_of_groups.append(AnswersOfAllInGroup(answers))
    return answers_of_groups


@click.command()
@click.argument("input_filename", type=str)
@click.option("--debug", type=bool, default=False, is_flag=True)
def cli(input_filename, debug):
    input_file = pathlib.Path(".") / input_filename
    with open(input_file) as open_file:
        puzzle_input = parse_input(open_file.read().splitlines())
    solution = solve(puzzle_input)
    print(
        f"The sum of the counts per group of questions to which "
        "[bold red]everyone[/bold red] "
        f"answered  'yes' is [bold green]{solution}[/bold green]."
    )


if __name__ == "__main__":
    cli()

```

</details>
</br>



### test_01.py

<details><summary>Code</summary>

```python
from solve_01 import parse_input, solve

TEST_INPUT = (
    """
abc

a
b
c

ab
ac

a
a
a
a

b
"""
).split("\n")


def test_solution():
    print(TEST_INPUT)
    result = solve(parse_input(TEST_INPUT))
    assert 11 == result

```

</details>
</br>



### test_02.py

<details><summary>Code</summary>

```python
from solve_02 import parse_input, solve

TEST_INPUT = (
    """abc

a
b
c

ab
ac

a
a
a
a

b"""
).split("\n")


def test_solution():
    print(TEST_INPUT)
    result = solve(parse_input(TEST_INPUT))
    assert 6 == result

```

</details>
</br>



</details>

</br>
<hr>
</br>


## Day 7



### solve_01.py

<details><summary>Code</summary>

```python
import logging
import pathlib
import re
from typing import List

import click
import networkx as nx  # type: ignore
from rich import print
from rich.logging import RichHandler


def solve(graph: nx.Graph, my_bag: str = "shiny gold") -> int:
    connected_bags = nx.algorithms.dag.ancestors(graph, my_bag)
    return len(connected_bags)


def parse_input(
    input_lines: List[str],
) -> nx.DiGraph:
    G = nx.DiGraph()
    LINE_PATTERN = r"(\d)* *(\w+ \w+) bag"
    for line in input_lines:
        logging.debug(line)
        matches = re.findall(LINE_PATTERN, line)
        parent = matches[0][1]
        edges = [
            (parent, match[1], int(match[0]))
            for match in matches[1:]
            if match[1] != "no other"
        ]
        logging.debug(edges)
        G.add_weighted_edges_from(edges, weight="amount")
    return G


@click.command()
@click.argument("input_filename", type=str)
@click.option("--debug", type=bool, default=False, is_flag=True)
def cli(input_filename, debug):
    BAG_TYPE = "shiny gold"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, handlers=[RichHandler()])

    input_file = pathlib.Path(".") / input_filename
    logging.info(f"Reading puzzle input from {input_file}.")
    with open(input_file) as open_file:
        input_lines = open_file.read().splitlines()
        puzzle_input = parse_input(input_lines)

    logging.info("Parsed puzzle input.")
    logging.info("Calculating solution...")
    solution = solve(puzzle_input)

    print(
        f"My {BAG_TYPE} bag can be stored in "
        f"[bold green]{solution}[/bold green] other bags."
    )


if __name__ == "__main__":
    cli()

```

</details>
</br>



### solve_02.py

<details><summary>Code</summary>

```python
import logging
import math
import pathlib
import re
from typing import List

import click
import networkx as nx  # type: ignore
from rich import print
from rich.logging import RichHandler


def solve(graph: nx.DiGraph, my_bag: str = "shiny gold") -> int:
    connected_bags = list(nx.algorithms.dag.descendants(graph, my_bag))
    amounts = [
        math.prod(
            [
                graph[start][end]["amount"]
                for start, end in zip(path[:-1], path[1:])
            ]
        )
        for node in connected_bags
        for path in nx.all_simple_paths(graph, my_bag, node)
    ]

    return sum(amounts)


def parse_input(
    input_lines: List[str],
) -> nx.DiGraph:
    G = nx.DiGraph()
    LINE_PATTERN = r"(\d)* *([a-z]+ [a-z]+) bag"
    for line in input_lines:
        logging.debug(line)
        matches = re.findall(LINE_PATTERN, line)
        parent = matches[0][1]
        edges = [
            (parent, match[1], int(match[0]))
            for match in matches[1:]
            if match[1] != "no other"
        ]
        logging.debug(edges)
        G.add_weighted_edges_from(edges, weight="amount")
    return G


@click.command()
@click.argument("input_filename", type=str)
@click.option("--debug", type=bool, default=False, is_flag=True)
def cli(input_filename, debug):
    BAG_TYPE = "shiny gold"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, handlers=[RichHandler()])
    input_file = pathlib.Path(".") / input_filename

    logging.info(f"Reading puzzle input from {input_file}.")
    with open(input_file) as open_file:
        input_lines = open_file.read().splitlines()
        puzzle_input = parse_input(input_lines)

    logging.info("Parsed puzzle input.")
    logging.info("Calculating solution...")
    solution = solve(puzzle_input)

    print(
        f"My {BAG_TYPE} bag must store "
        f"[bold green]{solution}[/bold green] other bags."
    )


if __name__ == "__main__":
    cli()

```

</details>
</br>



### test_01.py

<details><summary>Code</summary>

```python
from solve_01 import parse_input, solve

TEST_INPUT = (
    """light red bags contain 1 bright white bag, 2 muted yellow bags.
dark orange bags contain 3 bright white bags, 4 muted yellow bags.
bright white bags contain 1 shiny gold bag.
muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.
shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
dark olive bags contain 3 faded blue bags, 4 dotted black bags.
vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
faded blue bags contain no other bags.
dotted black bags contain no other bags."""
).split("\n")


def test_solution():
    print(TEST_INPUT)
    result = solve(parse_input(TEST_INPUT))
    assert 4 == result

```

</details>
</br>



### test_02.py

<details><summary>Code</summary>

```python
from solve_02 import parse_input, solve

TEST_INPUT = (
    """light red bags contain 1 bright white bag, 2 muted yellow bags.
dark orange bags contain 3 bright white bags, 4 muted yellow bags.
bright white bags contain 1 shiny gold bag.
muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.
shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
dark olive bags contain 3 faded blue bags, 4 dotted black bags.
vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
faded blue bags contain no other bags.
dotted black bags contain no other bags."""
).split("\n")

TEST_INPUT_2 = """shiny gold bags contain 2 dark red bags.
dark red bags contain 2 dark orange bags.
dark orange bags contain 2 dark yellow bags.
dark yellow bags contain 2 dark green bags.
dark green bags contain 2 dark blue bags.
dark blue bags contain 2 dark violet bags.
dark violet bags contain no other bags.""".split(
    "\n"
)


def test_solution():
    assert 32 == solve(parse_input(TEST_INPUT))
    assert 126 == solve(parse_input(TEST_INPUT_2))

```

</details>
</br>



</details>

</br>
<hr>
</br>


## Day 8



### console.py

<details><summary>Code</summary>

```python
import enum
import re
from dataclasses import dataclass
from typing import List


class Operation(enum.Enum):
    acc = "acc"
    jmp = "jmp"
    nop = "nop"


@dataclass
class Instruction:
    operation: Operation
    argument: int

    @classmethod
    def from_string(cls, string: str):
        INSTRUCTION_REGEX = r"^(nop|acc|jmp) ([+-]\d+)$"
        match = re.match(INSTRUCTION_REGEX, string)
        if match is None:
            raise ValueError("String does not match the instruction format")
        return cls(
            operation=Operation(match.group(1)), argument=int(match.group(2))
        )


class Operator:
    instructions: List[Instruction]
    accumulator: int
    offset: int

    def __init__(self, instructions: List[Instruction]) -> None:
        self.accumulator = 0
        self.offset = 0
        self.instructions = instructions

    def _execute(self, instruction: Instruction) -> None:
        operation_to_func = {
            Operation.acc: self._acc,
            Operation.jmp: self._jmp,
            Operation.nop: self._nop,
        }
        operation_to_func[instruction.operation](instruction.argument)

    def _acc(self, value) -> None:
        self.accumulator += value
        self.offset += 1

    def _jmp(self, value) -> None:
        self.offset += value

    def _nop(self, value) -> None:
        self.offset += 1

    def run(self) -> None:
        instructions_completed = []
        while self.offset not in instructions_completed:
            instructions_completed.append(self.offset)
            instruction = self.instructions[self.offset]
            print(instruction, self.offset, self.accumulator)
            self._execute(instruction)

    def validate_instructions(self) -> bool:
        instructions_completed = []
        while self.offset not in instructions_completed:
            instructions_completed.append(self.offset)
            instruction = self.instructions[self.offset]
            self._execute(instruction)
            if self.offset >= len(self.instructions):
                return True
        return False

```

</details>
</br>



### solve_01.py

<details><summary>Code</summary>

```python
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

```

</details>
</br>



### solve_02.py

<details><summary>Code</summary>

```python
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

```

</details>
</br>



### test_01.py

<details><summary>Code</summary>

```python
from solve_01 import parse_input, solve

TEST_INPUT = (
    """nop +0
acc +1
jmp +4
acc +3
jmp -3
acc -99
acc +1
jmp -4
acc +6"""
).split("\n")


def test_solution():
    instructions = parse_input(TEST_INPUT)
    print(instructions)

    result = solve(parse_input(TEST_INPUT))
    assert 5 == result

```

</details>
</br>



### test_02.py

<details><summary>Code</summary>

```python
from solve_02 import parse_input, solve

TEST_INPUT = (
    """nop +0
acc +1
jmp +4
acc +3
jmp -3
acc -99
acc +1
jmp -4
acc +6"""
).split("\n")


def test_solution():
    instructions = parse_input(TEST_INPUT)
    print(instructions)

    result = solve(parse_input(TEST_INPUT))
    assert 8 == result

```

</details>
</br>



</details>

</br>
<hr>
</br>
