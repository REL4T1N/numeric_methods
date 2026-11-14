import math
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def zeroes(n, m):
    # нулевая матрица n*m
    return [[0.0] * m for _ in range(n)]


def ident_matrix(n):
    # единичная матрица n*n
    A = zeroes(n, n)
    for i in range(n):
        A[i][i] = 1.0
    return A


def copy_matrix(A):
    return [row[:] for row in A]


def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def mul(A, B):
    p = len(B)
    return [[sum(A[i][r] * B[r][j] for r in range(p)) for j in range(len(B[0]))] for i in range(len(A))]


def is_symmetric(A, tol=1e-9):
    n = len(A)
    return all(abs(A[i][j] - A[j][i]) <= tol for i in range(n) for j in range(n))


def max_off_diag(A):
    # поиск индексов максимального по модулю не диагонального элемента
    n = len(A)
    p, q = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > abs(A[p][q]):
                p, q = i, j
    return p, q


def t_norm(A):
    n = len(A)
    return math.sqrt(sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j))


def build_U(n, p, q, phi):
    U = ident_matrix(n)
    c, s = math.cos(phi), math.sin(phi)
    U[p][p] = c
    U[q][q] = c
    U[p][q] = -s
    U[q][p] = s
    return U


def format_matrix(A):
    return "\n".join(["\t".join(map(str, row)) for row in A])


def jacobi(A0, eps):
    A = copy_matrix(A0)
    n = len(A)
    V = ident_matrix(n)

    if not is_symmetric(A):
        raise ValueError("Исходная матрица не симметрична!")
    else:
        print("Матрица A^0 = A симметрична: True")

    k = 0
    history = [(0, t_norm(A))]

    while True:
        t = t_norm(A)
        if t <= eps:
            break

        p, q = max_off_diag(A)
        apq, app, aqq = A[p][q], A[p][p], A[q][q]

        phi = 0.5 * math.atan2(2 * apq, app - aqq) # Вычисление угла поворота

        U = build_U(n, p, q, phi)
        A = mul(mul(transpose(U), A), U)
        V = mul(V, U)
        k += 1
        t = t_norm(A)

        # Проверка симметричности на каждой итерации через макс. отклонение A[i][j] от A[j][i]
        # asym = max(abs(A[i][j] - A[j][i]) for i in range(n) for j in range(n))
        # assert asym <= 1e-9, f"A^{k} не симметрична! max|a_ij - a_ji| = {asym}"
        history.append((k, t))
        
        # print(f"\nИтерация {k}: зануляем a_{p + 1}{q + 1} = {apq}")
        # print(f"phi = {phi:.6f}")
        # print("Матрица U:")
        # print(format_matrix(U))
        # print(f"A^{k} = U^T * A^{k - 1} * U:")
        # print(format_matrix(A))
        # print(f"Проверка симметричности A^{k}: max|a_ij - a_ji| = {asym:.2e} "
        #         f"{'<=' if asym <= 1e-9 else '>'} 1e-9 -> симметрична")
        # print(f"t(A^{k}) = {t:.6f} {'>=' if t > eps else '<'} eps = {eps}")

    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, V, history


# График зависимости погрешности t(A) от числа итераций k
def plot_error(history, out_path):

    ks = [k for k, _ in history]
    ts = [t for _, t in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, ts, marker="o")

    ax.set_xlabel("Номер итерации k")
    ax.set_ylabel("Погрешность t(A^k)")
    ax.set_title("Зависимость погрешности от числа итераций (метод Якоби)")

    # Масштаб: 1 клетка = 1
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True)

    fig.savefig(out_path, dpi=150)
    print(f"\nГрафик сохранён в {out_path}")



def main():
    A = [
        [0, -7, 7],
        [-7, -9, -5],
        [7, -5, -1],
    ]

    eps = 0.001

    print("Метод вращений Якоби (нахождение СЗ)")
    print("Исходная матрица A:")
    print(format_matrix(A))
    print(f"eps = {eps}")

    eigenvalues, V, history = jacobi(A, eps)

    print(f"\nВсего итераций: {len(history) - 1}")
    print("Собственные значения (диагональ итоговой матрицы):")
    for i, lam in enumerate(eigenvalues):
        print(f"  lambda{i + 1} = {lam:.6f}")

    # Собственные векторы — столбцы матрицы V = U^1 * U^2 * ... * U^k
    # (печатаем транспонированную, каждый вектор — строкой)
    print("\nСобственные векторы (столбцы матрицы V, по строкам):")
    print(format_matrix(transpose(V)))

    # Проверка ортогональности: скалярные произведения СВ попарно
    print(f"\nПроверка ортогональности собственных векторов (eps = {eps}):")
    n = len(A)
    orthogonal = True
    for i in range(n):
        for j in range(i + 1, n):
            dot = sum(V[r][i] * V[r][j] for r in range(n))
            ok = abs(dot) < eps
            orthogonal = orthogonal and ok
            print(f"  (v{i + 1}, v{j + 1}) = {dot:.2e} "
                  f"{'<' if ok else '>='} eps -> {'ортогональны' if ok else 'НЕ ортогональны'}")
    print(f"Собственные векторы ортогональны: {orthogonal}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab1-4.png")
    plot_error(history, out_path)


if __name__ == "__main__":
    main()
