import numpy as np
import numpy.typing as npt

BASE_3 = 3 ** np.arange(5)


class Correctness:
    INCORRECT = 0
    MISPLACED = 1
    CORRECT = 2

    def __init__(self, *correctness):
        self.__check_initializer_argument(*correctness)
        self._value = correctness

    def __eq__(self, other):
        if not isinstance(other, Correctness):
            return False
        return self._value == other._value

    @classmethod
    def __check_initializer_argument(cls, *correctness):
        """
        입력 결과를 검증하는 함수.
        """
        if len(correctness) != 5:
            raise ValueError("Correctness는 5개의 입력으로 이뤄져야 합니다.")
        if not all(
            c in [cls.INCORRECT, cls.MISPLACED, cls.CORRECT] for c in correctness
        ):
            raise ValueError(
                f"각 위치의 Correctness는 {cls.INCORRECT}, {cls.MISPLACED}, {cls.CORRECT} 중 하나여야 합니다."
            )

    @staticmethod
    def from_str(correctness_str: str) -> "Correctness":
        return Correctness(*map(int, correctness_str))

    @staticmethod
    def from_guess_and_answer(guess: str, answer: str) -> "Correctness":
        """
        guess와 answer를 비교하여 Correctness를 계산합니다.

        Wordle 규칙:
        1. 정확한 위치에 정확한 글자 -> CORRECT (2)
        2. 글자는 있지만 위치가 틀림 -> MISPLACED (1)
        3. 글자가 없음 -> INCORRECT (0)

        주의: 같은 글자가 여러 개 있을 경우, CORRECT를 먼저 처리하고
        남은 글자에 대해서만 MISPLACED를 처리합니다.
        """
        if len(guess) != 5 or len(answer) != 5:
            raise ValueError("guess와 answer는 모두 5글자여야 합니다.")

        result = [Correctness.INCORRECT] * 5
        answer_chars: list[str | None] = list(answer)

        # 1단계: CORRECT 처리 (정확한 위치)
        for i in range(5):
            if guess[i] == answer[i]:
                result[i] = Correctness.CORRECT
                answer_chars[i] = None  # 사용된 글자 표시

        # 2단계: MISPLACED 처리 (위치는 틀렸지만 글자는 존재)
        for i in range(5):
            if result[i] == Correctness.CORRECT:
                continue

            if guess[i] in answer_chars:
                result[i] = Correctness.MISPLACED
                # 해당 글자를 answer_chars에서 제거 (한 번만 사용)
                answer_chars[answer_chars.index(guess[i])] = None

        return Correctness(*result)

    @staticmethod
    def from_encoded_guess_and_answer(
        guess: npt.NDArray[np.uint8], answer: npt.NDArray[np.uint8]
    ) -> "Correctness":
        """
        numpy array로 인코딩된 guess와 answer를 비교하여 Correctness를 계산합니다.

        벡터화된 연산을 사용하여 from_guess_and_answer보다 빠르게 동작합니다.
        """
        if len(guess) != 5 or len(answer) != 5:
            raise ValueError("guess와 answer는 모두 길이 5여야 합니다.")

        result = np.zeros(5, dtype=np.uint8)
        answer_copy = answer.copy()

        # 1단계: CORRECT 처리 (정확한 위치)
        correct_mask = guess == answer
        result[correct_mask] = Correctness.CORRECT
        answer_copy[correct_mask] = (
            255  # 사용된 위치를 무효값으로 표시 (255는 알파벳 범위 밖)
        )

        # 2단계: MISPLACED 처리 (위치는 틀렸지만 글자는 존재)
        for i in range(5):
            if result[i] == Correctness.CORRECT:
                continue

            # answer_copy에서 guess[i]와 같은 값을 찾음
            matches = np.where(answer_copy == guess[i])[0]
            if len(matches) > 0:
                result[i] = Correctness.MISPLACED
                # 첫 번째 매칭된 위치를 무효화
                answer_copy[matches[0]] = 255

        return Correctness(*result)

    @staticmethod
    def compute_correctness_ids_batch(
        encoded_guess: npt.NDArray[np.uint8],  # shape (5,)
        encoded_answers: npt.NDArray[np.uint8],  # shape (n, 5)
    ) -> npt.NDArray[np.int32]:
        """
        하나의 guess에 대해 여러 answer들의 correctness id를 한 번에 계산.

        Wordle correctness 규칙을 완전히 numpy로 구현한 배치 버전.
        Correctness 객체를 생성하지 않고, 0/1/2 패턴을 직접 만든 뒤
        BASE_3로 id를 계산한다.

        규칙:
            - 2: CORRECT (자리/문자 모두 일치, "초록")
            - 1: MISPLACED (문자는 포함되어 있으나 자리 불일치, "노랑")
            - 0: INCORRECT ("회색")

        Args:
            encoded_guess: (5,) 배열 - 추측 단어
            encoded_answers: (n, 5) 배열 - 비교할 답 후보들

        Returns:
            (n,) 배열 - 각 answer에 대한 correctness id (0~242)
        """
        answers = encoded_answers
        n = answers.shape[0]

        # pattern: (n, 5)  / 0=INCORRECT, 1=MISPLACED, 2=CORRECT
        pattern = np.zeros_like(answers, dtype=np.uint8)

        # --------------------------------------------------------------------------------
        # 1) CORRECT(초록) 처리
        # --------------------------------------------------------------------------------
        # greens[i, j] == True -> answers[i, j] == guess[j]
        greens = answers == encoded_guess[None, :]  # shape (n, 5)
        pattern[greens] = Correctness.CORRECT  # 2

        # CORRECT 위치는 이후 노랑 계산에서 제외해야 하므로 마스킹
        # answers_masked에서 CORRECT 위치 문자를 -1로 바꿔 "이미 소비된 문자"로 처리
        answers_masked = answers.astype(np.int16).copy()
        answers_masked[greens] = -1

        # --------------------------------------------------------------------------------
        # 2) 각 answer별로 남은 문자 개수 계산 (CORRECT 제외)
        #    remaining_counts[i, c] = answers[i]에서 문자 c가 남은 개수
        # --------------------------------------------------------------------------------
        remaining_counts = np.zeros((n, 26), dtype=np.int16)

        # 알파벳 26개에 대해 카운트 (상수 루프)
        for letter in range(26):
            # answers_masked == letter인 위치 개수를 axis=1 방향으로 세면
            # 각 answer별 letter 개수가 나옴
            remaining_counts[:, letter] = np.sum(answers_masked == letter, axis=1)

        # --------------------------------------------------------------------------------
        # 3) MISPLACED(노랑) 처리
        #    CORRECT로 이미 배정된 것은 제외하고,
        #    남은 remaining_counts를 소모해 가며 노랑을 채운다.
        #    Wordle 규칙상 위치 순서대로 처리해도 일관된 결과가 나온다.
        # --------------------------------------------------------------------------------
        for pos in range(5):
            letter = int(encoded_guess[pos])

            # 이 위치가 CORRECT인 행은 노랑 후보에서 제외
            not_green = ~greens[:, pos]

            # letter가 남아 있는 answer들만 노랑이 될 수 있음
            has_letter_left = remaining_counts[:, letter] > 0

            # 동시에 만족하는 행들만 이 위치에서 MISPLACED가 된다.
            yellow_mask = not_green & has_letter_left
            if not np.any(yellow_mask):
                continue

            # 해당 위치를 MISPLACED로 표시하고
            pattern[yellow_mask, pos] = Correctness.MISPLACED  # 1

            # 그 answer들의 remaining_counts에서 이 문자 1개씩 소모
            remaining_counts[yellow_mask, letter] -= 1

        # --------------------------------------------------------------------------------
        # 4) 3진법 코드 -> correctness id로 변환
        # --------------------------------------------------------------------------------
        # pattern: (n, 5)  / BASE_3: (5,)
        # id[i] = Σ_j pattern[i, j] * BASE_3[j]
        pattern_int = pattern.astype(np.int32)
        correctness_ids = (pattern_int * BASE_3).sum(axis=1, dtype=np.int32)

        return correctness_ids

    @staticmethod
    def from_ndarray(correctness: npt.NDArray[np.uint8]) -> "Correctness":
        return Correctness(*correctness)

    @staticmethod
    def from_id(id: int) -> "Correctness":
        return Correctness(*[id // (3**i) % 3 for i in range(5)])

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return self._value

    def as_ndarray(self) -> npt.NDArray[np.uint8]:
        return np.array(self._value)

    def id(self) -> int:
        """
        3진법 기반으로 id 생성하는 함수
        예) [0, 1, 2, 1, 0] -> 48, [1, 0, 0, 0, 0] -> 1
        """
        return np.dot(self.as_ndarray(), BASE_3)

    def is_all_correct(self) -> bool:
        return all(c == Correctness.CORRECT for c in self._value)
