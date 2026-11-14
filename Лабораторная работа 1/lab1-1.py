import numpy as np


# Решение СЛАУ через LU: прямой + обратный ход
def find_x(L, U, b):
    """Решение Lz = b (прямой ход), затем Ux = z (обратный ход)."""
    n = len(b)
    z = [0] * n
    for i in range(n):
        z_sum = 0
        for j in range(i):
            z_sum += L[i][j] * z[j]
        z[i] = b[i] - z_sum

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x_sum = 0
        for j in range(i + 1, n):
            x_sum += U[i][j] * x[j]
        x[i] = (z[i] - x_sum) / U[i][i]
    return x


# LU-разложение с выбором главного элемента
def lu_decomposition(A):
    """Разложение матрицы A на P, L, U такие, что PA = LU.

    Возвращает:
        p — вектор перестановок
        L — нижнетреугольная матрица с единицами на диагонали
        U — верхнетреугольная матрица
    """
    n = len(A)
    U = [row.copy() for row in A]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    p = list(range(n))
    num_swaps = 0

    for k in range(n - 1):
        # Выбор главного элемента: ищем строку с max|U[i][k]| от k до n-1
        max_index = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_index = i

        # перестановка строк (если pivot не на месте)
        if max_index != k:
            U[k], U[max_index] = U[max_index], U[k]
            L[k][:k], L[max_index][:k] = L[max_index][:k], L[k][:k]
            p[k], p[max_index] = p[max_index], p[k]
            num_swaps += 1

        # вычисляем множители и зануляем элементы ниже pivot
        for i in range(k + 1, n):
            mu = U[i][k] / U[k][k]
            L[i][k] = mu
            for j in range(k, n):
                U[i][j] -= mu * U[k][j]

    return p, L, U, num_swaps


# Определитель
def determinant(U, num_swaps):
    """det(A) = (-1)^(число перестановок) * произведение диагональных элементов U."""
    det = (-1) ** num_swaps
    for i in range(len(U)):
        det *= U[i][i]
    return det


# Обратная матрица 
def inverse(p, L, U, n):
    """Находит A^(-1), решая Ax_j = e_j для каждого столбца единичной матрицы."""
    A_inv = [[0.0] * n for _ in range(n)]
    for col in range(n):
        e = [0] * n
        e[col] = 1
        e_mod = [e[p[i]] for i in range(n)]
        x_col = find_x(L, U, e_mod)
        for i in range(n):
            A_inv[i][col] = x_col[i]
    return A_inv


# Решение СЛАУ
def solve_system(p, L, U, b):
    """Решает СЛАУ Ax = b через LU-разложение (применяет перестановки и вызывает find_x)."""
    n = len(b)
    b_mod = [b[p[i]] for i in range(n)]
    return find_x(L, U, b_mod)


def main():
    A = [
        [3, -8, 1, -7],
        [6, 4, 8, 5],
        [-1, 1, -9, -3],
        [-6, 6, 9, -4],
    ]
    b = [96, -13, -54, 82]
    n = len(A)

    A = [[float(x) for x in row] for row in A]

    # LU-разложение
    p, L, U, num_swaps = lu_decomposition(A)

    # Решение СЛАУ
    x = solve_system(p, L, U, b)

    # Определитель
    det = determinant(U, num_swaps)

    # Обратная матрица (существует только при det != 0)
    A_inv = inverse(p, L, U, n)

    x_rounded = [round(v, 10) for v in x]
    print("Значения x:", x_rounded)
    print("Определитель:", round(det, 10))

    A_inv_np = np.array(A_inv)
    print("Обратная матрица:\n", A_inv_np)

    # Проверка
    print("\nПроверка A*x = b:")
    print(np.matmul(A, x))
    print("Проверка A*A_inv = I:")
    print(np.round(np.matmul(A, A_inv_np), 8))


if __name__ == "__main__":
    main()
