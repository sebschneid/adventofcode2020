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
