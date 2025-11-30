import sys
import time
from typing import Callable

print = sys.stdout.write


def log_with_time(filter: Callable | None = None):
    def deco(func: Callable):
        def wrapper(*args):
            start_time = time.time_ns()
            result = func(*args)
            end_time = time.time_ns()

            if filter is None or filter(result):
                print(
                    f"{func.__name__}{args[1:]} = {result} ({(end_time - start_time) / 1_000_000} ms)\n"
                )

            return result

        return wrapper

    return deco
