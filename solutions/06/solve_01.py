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
