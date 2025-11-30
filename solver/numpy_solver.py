import numpy as np
import numpy.typing as npt

from correctness import Correctness
from history import History
from solver.solver import Solver
from time_logger import log_with_time


class NumpySolver(Solver):
    # "tares"를 미리 인코딩해 둔 상수
    FIRST_GUESS_ENCODED = np.array([19, 0, 17, 4, 18], dtype=np.uint8)

    def __init__(self, hard_code_first_guess: bool = False):
        """
        Args:
            hard_code_first_guess:
                True  => 첫 번째 추측을 하드코딩된 최적 단어("tares")로 사용
                False => 첫 번째 추측도 엔트로피 계산으로 선택
        """
        self.hard_code_first_guess = hard_code_first_guess

        # 초기화 시점에 채워지는 필드들 (type 힌트용)
        self.word_list: list[str] = []
        self.word_to_index: dict[str, int] = {}
        self.encoded_word_list: npt.NDArray[np.uint8]  # (N, 5)
        self.encoded_answer_candidates: npt.NDArray[np.uint8]  # (M, 5)
        self.answer_indices: npt.NDArray[np.int32]  # 길이 M
        self._is_first_try: bool = True

    # --------------------------------------------------------------------------------
    # 추상 메서드 구현
    # --------------------------------------------------------------------------------

    def initialize(self, word_list: list[str]) -> None:
        """
        Solver 공통 인터페이스 구현.
        전체 단어 리스트를 인코딩하고, 초기 후보 집합을 전체 단어로 설정.
        """
        # 원본 문자열 리스트 저장
        self.word_list = word_list[:]
        # 단어 -> 인덱스 매핑 (O(1)으로 인덱스 찾기 위함)
        self.word_to_index = {w: i for i, w in enumerate(self.word_list)}

        # 문자열을 numpy array로 인코딩 (a=0, b=1, ..., z=25)
        self.encoded_word_list = self._encode_words(self.word_list)  # (N, 5)

        # 초기에는 모든 단어가 정답 후보
        n = len(self.word_list)
        self.answer_indices = np.arange(n, dtype=np.int32)
        self.encoded_answer_candidates = self.encoded_word_list.copy()

        # 첫 트라이 플래그 초기화
        self._is_first_try = True

    def guess_word(self) -> str:
        """
        현재까지의 힌트를 바탕으로 다음에 시도할 단어를 반환.
        """
        # 첫 추측을 하드코딩된 단어로 사용할 옵션
        if self._is_first_try and self.hard_code_first_guess:
            self._is_first_try = False
            return self._decode_word(self.FIRST_GUESS_ENCODED)

        self._is_first_try = False

        # 후보가 1개로 확정되었다면 그 단어를 바로 반환
        if self.has_solved():
            # answer_indices 를 이용해 원본 word_list에서 가져옴
            solved_idx = int(self.answer_indices[0])
            return self.word_list[solved_idx]

        # 모든 단어에 대한 엔트로피 점수 계산
        entropy_scores = self._get_entropy_scores()

        # 최대 엔트로피를 주는 단어 선택
        best_idx = int(np.argmax(entropy_scores))
        return self._decode_word(self.encoded_word_list[best_idx])

    def analyze_history(self, history: History) -> None:
        """
        방금 시도한 guess와 correctness(History)를 바탕으로
        가능한 정답 후보 집합을 필터링.
        """
        self.encoded_answer_candidates = self._filter_candidates_by_history(history)

    def has_solved(self) -> bool:
        """
        정답 후보가 정확히 하나만 남았는지 여부.
        """
        return len(self.encoded_answer_candidates) == 1

    # --------------------------------------------------------------------------------
    # 헬퍼 메서드
    # --------------------------------------------------------------------------------

    def _encode_words(self, words: list[str]) -> npt.NDArray[np.uint8]:
        """
        문자열 리스트를 numpy array로 변환.
        각 문자는 a=0, b=1, ..., z=25로 인코딩됨.

        예: ["hello", "world"] -> [[7, 4, 11, 11, 14], [22, 14, 17, 11, 3]]
        """
        n = len(words)
        encoded = np.zeros((n, 5), dtype=np.uint8)

        for i, word in enumerate(words):
            # frombuffer로 5글자를 한 번에 처리
            encoded[i] = np.frombuffer(word.encode("ascii"), dtype=np.uint8) - ord("a")

        return encoded

    def _decode_word(self, encoded_word: npt.NDArray[np.uint8]) -> str:
        """
        인코딩된 단어를 문자열로 변환.

        예: [7, 4, 11, 11, 14] -> "hello"
        """
        return "".join(chr(int(c) + ord("a")) for c in encoded_word)

    def _get_entropy_scores(self) -> npt.NDArray[np.float64]:
        """
        모든 단어에 대한 엔트로피 점수를 계산.
        (현재는 전체 단어를 guess 후보로 사용)
        """
        n_words = len(self.encoded_word_list)
        scores = np.zeros(n_words, dtype=np.float64)

        # 현재 정답 후보 (encoded)
        # self.encoded_answer_candidates 와 self.answer_indices는 항상 동기화된 상태
        for i in range(n_words):
            encoded_guess = self.encoded_word_list[i]
            scores[i] = self._get_entropy_score(encoded_guess)

        return scores

    @log_with_time(filter=lambda x: x > 0)
    def _get_entropy_score(self, encoded_guess: npt.NDArray[np.uint8]) -> float:
        """
        특정 guess에 대한 엔트로피 점수 계산.

        엔트로피 = -Σ(p * log2(p))
        여기서 p는 각 correctness 패턴이 나올 확률
        """
        n_candidates = len(self.encoded_answer_candidates)
        if n_candidates == 0:
            return 0.0

        # Correctness 클래스의 배치 계산 메서드 사용
        correctness_ids = Correctness.compute_correctness_ids_batch(
            encoded_guess, self.encoded_answer_candidates
        )

        # correctness id는 0~242 범위 (3^5 = 243 가지)
        counts = np.bincount(correctness_ids, minlength=243)
        counts = counts[counts > 0]  # 0인 값 제거

        # 엔트로피 계산: -Σ(p * log2(p))
        probabilities = counts / n_candidates
        entropy = -np.dot(probabilities, np.log2(probabilities))

        return float(entropy)

    def _filter_candidates_by_history(self, history: History) -> npt.NDArray[np.uint8]:
        """
        히스토리를 기반으로 가능한 답을 필터링.

        Returns:
            필터링된 인코딩된 단어들 (m, 5) 배열
        """
        # guess 단어를 다시 인코딩하지 않고, 미리 만들어 둔 매핑을 통해 인덱스로 접근
        # (재인코딩을 피해서 약간의 성능 이득)
        guess_idx = self.word_to_index[history.guessed_word]
        encoded_guess = self.encoded_word_list[guess_idx]

        target_correctness_id = history.correctness.id()

        # Correctness 클래스의 배치 계산 메서드 사용
        correctness_ids = Correctness.compute_correctness_ids_batch(
            encoded_guess, self.encoded_answer_candidates
        )

        # 정답 후보와 동일한 correctness 패턴을 만드는 단어만 남김
        mask = correctness_ids == target_correctness_id

        # answer_indices도 같이 필터링해서, 문자열/인덱스 양쪽을 모두 추적
        self.answer_indices = self.answer_indices[mask]

        # encoded 후보 배열도 필터링
        filtered_encoded = self.encoded_answer_candidates[mask]

        return filtered_encoded
