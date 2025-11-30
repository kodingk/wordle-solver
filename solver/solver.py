import abc

from history import History


class Solver(abc.ABC):
    """
    기본 wordle에서 다음 단어 추론하는 기능을 제공하는 추상 클래스
    """

    @abc.abstractmethod
    def guess_word(self) -> str:
        """
        다음 best 단어를 반환.
        """
        pass

    @abc.abstractmethod
    def initialize(self, word_list: list[str]) -> None:
        """
        solver 인스턴스를 완전히 초기화하는 함수.
        """
        pass

    @abc.abstractmethod
    def analyze_history(self, history: History) -> None:
        """
        새 결과를 토대로 다음 guess_word 콜을 위해 대기하는 함수
        """
        pass

    @abc.abstractmethod
    def has_solved(self) -> bool:
        """
        solver 인스턴스가 문제를 해결했는지 여부를 반환.
        """
        pass
