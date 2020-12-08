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
