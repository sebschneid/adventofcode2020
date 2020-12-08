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
