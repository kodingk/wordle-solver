import argparse

from game import Game
from solver.numpy_solver import NumpySolver
from solver.pure_pythonic_solver import PurePythonicSolver

WORD_LIST_PATH = "dictionary.txt"


def load_word_list(file_path=WORD_LIST_PATH):
    try:
        with open(file_path, "r") as file:
            return [line.strip().lower() for line in file.readlines()]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wordle Solver")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast NumPy-based solver (default: Pure Python solver)",
    )
    parser.add_argument(
        "--no-hardcode",
        action="store_true",
        help="Calculate first guess using entropy instead of hardcoded 'tares'",
    )
    args = parser.parse_args()

    game = Game(load_word_list())

    hard_code_first_guess = not args.no_hardcode

    if args.fast:
        print(
            f"Using NumpySolver (fast mode, hard_code_first_guess={hard_code_first_guess})"
        )
        solver = NumpySolver(hard_code_first_guess=hard_code_first_guess)
    else:
        print(
            f"Using PurePythonicSolver (hard_code_first_guess={hard_code_first_guess})"
        )
        solver = PurePythonicSolver(hard_code_first_guess=hard_code_first_guess)

    game.run(solver)
