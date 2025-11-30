# Wordle Solver

엔트로피 기반 알고리즘을 사용하는 Wordle 게임 솔버입니다.

## 특징

- 🎯 **엔트로피 기반 알고리즘**: 정보 이론을 활용하여 최적의 추측 단어 선택
- ⚡ **두 가지 구현 방식**:
  - **Pure Python Solver**: 순수 Python으로 구현된 버전
  - **NumPy Solver**: NumPy 벡터화 연산을 활용한 고속 버전
- 🔧 **유연한 설정**: 첫 번째 추측을 하드코딩하거나 계산하도록 선택 가능

## 요구사항

```bash
pip install numpy
```

## 사용법

### 기본 실행 (Pure Python Solver)

```bash
python main.py
```

### 고속 모드 (NumPy Solver)

```bash
python main.py --fast
```

### 첫 추측도 엔트로피로 계산

```bash
python main.py --no-hardcode
```

### 옵션 조합

```bash
python main.py --fast --no-hardcode
```

## 게임 진행 방법

1. 프로그램이 추천하는 단어를 Wordle 게임에 입력합니다.
2. 게임 결과를 숫자로 입력합니다:
   - `0`: 회색 (INCORRECT) - 해당 문자가 답에 없음
   - `1`: 노란색 (MISPLACED) - 해당 문자가 답에 있지만 위치가 틀림
   - `2`: 초록색 (CORRECT) - 해당 문자가 정확한 위치에 있음

### 예시

```
계산 완료: tares
결과를 입력하세요 (예: 02011): 00120
계산 완료: clump
결과를 입력하세요 (예: 02011): 01020
계산 완료: pound
결과를 입력하세요 (예: 02011): 22222
게임 종료: pound
```

위 예시에서:
- `tares` → `00120`: t(회색), a(회색), r(초록), e(회색), s(회색)
- `clump` → `01020`: c(회색), l(노랑), u(회색), m(초록), p(회색)
- `pound` → `22222`: 정답!

## 커맨드 라인 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--fast` | NumPy 기반 고속 solver 사용 | False (Pure Python) |
| `--no-hardcode` | 첫 추측도 엔트로피 계산으로 선택 | False (하드코딩된 "tares" 사용) |

## 프로젝트 구조

```
wordle-solver/
├── main.py                     # 진입점
├── game.py                     # 게임 진행 로직
├── correctness.py              # Correctness 계산 로직
├── history.py                  # 추측 히스토리 관리
├── solver/
│   ├── solver.py               # Solver 추상 클래스
│   ├── pure_pythonic_solver.py # Pure Python 구현
│   └── numpy_solver.py         # NumPy 벡터화 구현
├── time_logger.py              # 성능 측정 데코레이터
└── dictionary.txt              # 단어 사전
```

## 알고리즘 설명

### 엔트로피 기반 단어 선택

각 단어에 대해 **정보 엔트로피(Information Entropy)**를 계산하여 가장 많은 정보를 얻을 수 있는 단어를 선택합니다.

```
엔트로피 = -Σ(p * log₂(p))
```

여기서 `p`는 각 correctness 패턴(0~2의 5자리 조합, 총 243가지)이 나올 확률입니다.

### Correctness 계산

Wordle의 규칙을 정확히 구현:
1. **CORRECT(초록)**: 정확한 위치에 정확한 문자
2. **MISPLACED(노랑)**: 문자는 있지만 위치가 틀림
3. **INCORRECT(회색)**: 문자가 답에 없음

중복 문자 처리:
- CORRECT를 먼저 배정한 후
- 남은 문자에 대해서만 MISPLACED 배정
- 각 문자는 한 번만 사용됨

### Pure Python vs NumPy

| 특징 | Pure Python | NumPy |
|------|-------------|-------|
| 구현 방식 | List, dict, Counter 사용 | NumPy 배열 및 벡터 연산 |
| Correctness 계산 | 루프 기반 | 완전 벡터화 |
| 메모리 사용 | 문자열 리스트 유지 | 인코딩된 정수 배열 (a=0, b=1, ..., z=25) |
| 속도 | 중간 | 빠름 (대량 데이터) |
| 코드 가독성 | 높음 | 중간 |

## 성능 최적화

### NumPy Solver의 주요 최적화

1. **배치 Correctness 계산**: `Correctness.compute_correctness_ids_batch()`
   - 여러 answer에 대해 한 번에 계산
   - 브로드캐스팅을 활용한 벡터 연산

2. **문자 빈도 추적**: `remaining_counts` (n, 26) 배열
   - 각 answer별 남은 문자 개수를 한 번에 계산

3. **인덱스 기반 관리**:
   - 문자열 대신 인덱스로 후보 추적
   - 재인코딩 방지

4. **bincount 사용**: O(n log n) → O(n)
   - `np.unique()` 대신 `np.bincount()` 사용
