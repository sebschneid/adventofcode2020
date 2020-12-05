from solve_02 import parse_input, solve

TEST_INPUT = ["1-3 a: abcde\n", "1-3 b: cdefg\n", "2-9 c: ccccccccc\n"]


def test_solution():
    result = solve(parse_input(TEST_INPUT))
    assert 1 == result
