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
