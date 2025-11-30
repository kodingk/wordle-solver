import math
from collections import Counter

from correctness import Correctness
from history import History
from solver.solver import Solver
from time_logger import log_with_time


class PurePythonicSolver(Solver):
    FIRST_GUESS = "tares"

    def __init__(self, hard_code_first_guess: bool = False):
        """
        Args:
            hard_code_first_guess: True면 첫 번째 추측을 하드코딩된 최적 단어로 사용,
                                   False면 엔트로피 계산으로 첫 단어도 선택
        """
        self.hard_code_first_guess = hard_code_first_guess

    # --------------------------------------------------------------------------------
    # 추상 메서드 구현
    # --------------------------------------------------------------------------------

    def initialize(self, word_list: list[str]) -> None:
        self.word_list = word_list[:]
        self.answer_candidates = word_list[:]
        self.answer = None
        # 첫 트라이의 경우 정해져 있는 최선의 단어를 하드코딩하여 반환.
        self.__is_first_try = True

    def guess_word(self) -> str:
        if self.__is_first_try and self.hard_code_first_guess:
            self.__is_first_try = False
            return self.FIRST_GUESS

        self.__is_first_try = False

        if self.has_solved():
            return self.answer_candidates[0]

        es = self._get_entropy_scores()
        return max(es.items(), key=lambda x: x[1])[0]

    def analyze_history(self, history: History) -> None:
        self.answer_candidates = self._filter_candidates_by_history(history)

    def has_solved(self) -> bool:
        return len(self.answer_candidates) == 1

    # --------------------------------------------------------------------------------
    # 헬퍼 메서드
    # --------------------------------------------------------------------------------

    def _get_entropy_scores(self) -> dict[str, float]:
        d = dict()
        for guess in self.word_list:
            d[guess] = self._get_entropy_score(guess)

        return d

    @log_with_time(filter=lambda x: x != 0)  # entropy가 0이 아닌 경우만 출력
    def _get_entropy_score(self, guess: str) -> float:
        total = len(self.answer_candidates)
        counted_values = list(
            Counter(
                map(
                    lambda answer: Correctness.from_guess_and_answer(
                        guess, answer
                    ).id(),
                    self.answer_candidates,
                )
            ).values()
        )

        return sum(map(lambda x: -(x / total) * math.log2((x / total)), counted_values))

    def _filter_candidates_by_history(self, history: History) -> list[str]:
        return list(
            filter(
                lambda a: Correctness.from_guess_and_answer(history.guessed_word, a)
                == history.correctness,
                self.answer_candidates,
            )
        )
