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
