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
