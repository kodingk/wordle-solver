from correctness import Correctness
from history import History
from solver.solver import Solver


class Game:
    def __init__(self, word_list: list[str]):
        self.__check_initializer_input(word_list)

        self.word_list: list[str] = word_list
        self.answer: str | None = None

    @classmethod
    def __check_initializer_input(cls, word_list: list[str]):
        def check_word(w: str):
            return len(w) == 5 and w.isalpha() and w.islower()

        if not all(check_word(w) for w in word_list):
            raise ValueError("잘못된 단어 리스트입니다.")

    def run(self, solver: Solver):
        solver.initialize(self.word_list)

        while True:
            guess = solver.guess_word()
            if solver.has_solved():
                self.answer = guess
                break

            print(f"계산 완료: {guess}")
            c = Correctness.from_str(input("결과를 입력하세요 (예: 02011): "))

            h = History(guess, c)
            solver.analyze_history(h)

        print(f"게임 종료: {self.answer}")
