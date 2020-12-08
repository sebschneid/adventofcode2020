import enum
import re
from dataclasses import dataclass
from typing import List


class Operation(enum.Enum):
    acc = "acc"
    jmp = "jmp"
    nop = "nop"


@dataclass
class Instruction:
    operation: Operation
    argument: int

    @classmethod
    def from_string(cls, string: str):
        INSTRUCTION_REGEX = r"^(nop|acc|jmp) ([+-]\d+)$"
        match = re.match(INSTRUCTION_REGEX, string)
        if match is None:
            raise ValueError("String does not match the instruction format")
        return cls(
            operation=Operation(match.group(1)), argument=int(match.group(2))
        )


class Operator:
    instructions: List[Instruction]
    accumulator: int
    offset: int

    def __init__(self, instructions: List[Instruction]) -> None:
        self.accumulator = 0
        self.offset = 0
        self.instructions = instructions

    def _execute(self, instruction: Instruction) -> None:
        operation_to_func = {
            Operation.acc: self._acc,
            Operation.jmp: self._jmp,
            Operation.nop: self._nop,
        }
        operation_to_func[instruction.operation](instruction.argument)

    def _acc(self, value) -> None:
        self.accumulator += value
        self.offset += 1

    def _jmp(self, value) -> None:
        self.offset += value

    def _nop(self, value) -> None:
        self.offset += 1

    def run(self) -> None:
        instructions_completed = []
        while self.offset not in instructions_completed:
            instructions_completed.append(self.offset)
            instruction = self.instructions[self.offset]
            print(instruction, self.offset, self.accumulator)
            self._execute(instruction)

    def validate_instructions(self) -> bool:
        instructions_completed = []
        while self.offset not in instructions_completed:
            instructions_completed.append(self.offset)
            instruction = self.instructions[self.offset]
            self._execute(instruction)
            if self.offset >= len(self.instructions):
                return True
        return False
