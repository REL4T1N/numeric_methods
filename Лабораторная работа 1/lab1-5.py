import math
import numpy as np


def zeroes(n, m):
    return [[0.0] * m for _ in range(n)]


def ident_matrix(n):
    A = zeroes(n, n)
    for i in range(n):
        A[i][i] = 1.0
    return A


def copy_matrix(A):
    return [row[:] for row in A]


def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def subtract(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def vector_norm(v):
    """Евклидова норма вектора"""
    return math.hypot(*v)


def sign(a):
    # при a == 0 выбираем знак +1.
    return -1 if a < 0 else 1


def format_matrix(A):
    return "\n".join("\t".join(f"{value:.6f}" for value in row) for row in A)


def format_vector(v):
    return "[" + ", ".join(f"{value:.6f}" for value in v) + "]"

def scalar_mul(A, coefficient):
    return [[coefficient * value for value in row] for row in A]

def householder_matrix(v):
    v_t = transpose(v)
    v_vt = mul(v, v_t)        # Матрица n×n
    vt_v = mul(v_t, v)[0][0]  # Число

    return subtract(
        ident_matrix(len(v)),
        scalar_mul(v_vt, 2 / vt_v),
    )

def qr_decomposition(A):
    """A = QR через отражения Хаусхолдера"""
    n = len(A)
    Q = ident_matrix(n)
    R = copy_matrix(A)

    for k in range(n - 1):
        norm = vector_norm([R[i][k] for i in range(k, n)])
        if norm == 0:
            continue
        v = zeroes(n, 1)
        v[k][0] = R[k][k] + sign(R[k][k]) * norm
        for i in range(k+1, n):
            v[i][0] = R[i][k]

        H = householder_matrix(v)

        print(f"Отражение {k + 1}, v = {format_vector([row[0] for row in v])}")
        print("H:\n" + format_matrix(H))

        R = mul(H, R) # A^(k+1) = H * A^k
        Q = mul(Q, H) # Q = П_k H^k

    return Q, R


def block_roots(A, j):
    """Корни для диагонального блока B размера 2x2."""
    a, b = A[j][j], A[j][j + 1]
    c, d = A[j + 1][j], A[j + 1][j + 1]
    trace = a + d
    discriminant = (a - d) ** 2 + 4 * b * c
    if discriminant >= 0:
        root = math.sqrt(discriminant)
    else:
        root = complex(0, math.sqrt(-discriminant))
    return [(trace + root) / 2, (trace - root) / 2]


def extract_eigenvalues(A, A_prev, eps):
    """Возвращает все СЗ при выполнении критериев, иначе None."""
    n = len(A)
    eigenvalues = []
    j = 0
    while j < n:
        # Вещественное СЗ: норма хвоста столбца ниже диагонали.
        tail = vector_norm([A[i][j] for i in range(j + 1, n)])
        if tail <= eps: # проверка вещественных корней
            eigenvalues.append(A[j][j])
            j += 1
            continue

        # Проверка наличия второго комплексного столбца
        # Проверка наличия предыдущей матрицы Ak и её комплексных корней
        if any(vector_norm([A[i][col] for i in range(j + 2, n)]) > eps for col in (j, j + 1)):
            return None
        roots = block_roots(A, j)
        if not isinstance(roots[0], complex) or A_prev is None:
            return None
        previous = block_roots(A_prev, j)
        if not isinstance(previous[0], complex):
            return None

        # \delta комплексных корней
        change = min(max(abs(roots[i] - previous[i]) for i in range(2)),
                     max(abs(roots[i] - previous[1 - i]) for i in range(2)))
        if change > eps: # Если условие выхода вещественных или комплексных корней не выполнилось, то продолжаем итерации
            return None
        eigenvalues.extend(roots) # Запись комплексных корней
        j += 2

    return eigenvalues


def qr_eigenvalues(A, eps):
    """QR без сдвигов: остановка только при сходимости всех СЗ."""

    Ak = copy_matrix(A)
    previous = None
    iterations = 0

    while True:
        eigenvalues = extract_eigenvalues(Ak, previous, eps)
        if eigenvalues is not None:
            return eigenvalues, Ak, iterations

        previous = copy_matrix(Ak)
        Q, R = qr_decomposition(Ak)
        Ak = mul(R, Q)
        iterations += 1
        print(f"\nИтерация {iterations}")
        print("Q:\n" + format_matrix(Q))
        print("R:\n" + format_matrix(R))
        print("A^k:\n" + format_matrix(Ak))


def main():
    A = [[5, 8, -2], [7, -2, -4], [5, 8, -1]]
    eps = 1e-3
    print("Исходная матрица:\n" + format_matrix(A))
    eigenvalues, Ak, iterations = qr_eigenvalues(A, eps)
    print("Количество итераций:", iterations)
    print("Полученная матрица A^k:\n" + format_matrix(Ak))
    print("Собственные значения:", format_vector(eigenvalues))

    eigenvalues_np = np.linalg.eigvals(np.array(A, dtype=float))
    print("СЗ NumPy:", format_vector(eigenvalues_np))

if __name__ == "__main__":
    main()
