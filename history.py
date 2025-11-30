from correctness import Correctness


class History:
    def __init__(self, guessed_word: str, correctness: Correctness):
        self.__check_initializer_argument(guessed_word, correctness)

        self.guessed_word = guessed_word
        self.correctness = correctness

    @classmethod
    def __check_initializer_argument(cls, guessed_word: str, correctness: Correctness):
        if (
            not guessed_word.isalpha()
            or not guessed_word.islower()
            or not len(guessed_word) == 5
        ):
            raise ValueError(f"단어 입력이 잘못되었습니다: {guessed_word}")
