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
