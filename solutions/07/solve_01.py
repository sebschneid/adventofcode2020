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
