import numpy as np


def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def subtract(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def zeroes(n, m=1):
    return [[0.0] * m for _ in range(n)]


def mul(A, B):
    return [[sum(A[i][r] * B[r][j] for r in range(len(A[0]))) for j in range(len(B[0]))] for i in range(len(A))]


def norm_vec(v):
    return sum(abs(v[i][0]) for i in range(len(v)))


def norm_mat(A):
    return max(sum(abs(A[i][j]) for j in range(len(A[0]))) for i in range(len(A)))


def copy(A):
    return [row.copy() for row in A]


def format_matrix(A):
    return "\n".join(["\t".join(map(str, row)) for row in A])


def format_vector(v):
    return "[" + ", ".join(f"{row[0]:.6f}" for row in v) + "]"


def poly_add(p, q):
    m = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0.0) + (q[i] if i < len(q) else 0.0) for i in range(m)]


def poly_mul(p, q):
    r = [0.0] * (len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            r[i + j] += p[i] * q[j]
    return r


def det_poly(M):
    # рекурсивный расчёт определителя через разложение первой строки
    n = len(M)
    if n == 1:
        return M[0][0]
    result = [0.0]
    for j in range(n):
        minor = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        term = poly_mul(M[0][j], det_poly(minor))
        if j % 2 == 1:
            term = [-c for c in term]
        result = poly_add(result, term)
    return result


def find_real_roots(coeffs, R, step=1e-3, eps=1e-12):
    # вычисление действительных корней
    def val(x):
        r = 0.0
        for c in coeffs:
            r = r * x + c
        return r

    roots = []
    x = -R
    while x < R:
        f1, f2 = val(x), val(x + step)
        if f1 == 0.0:
            roots.append(x)
        elif f1 * f2 < 0:
            lo, hi = x, x + step
            while hi - lo > eps:
                mid = (lo + hi) / 2
                if val(lo) * val(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        x += step
    return roots


def deflate(coeffs, r):
    # Деление многочлена на (lambda - r) Горнером
    q = [coeffs[0]]
    for c in coeffs[1:-1]:
        q.append(c + q[-1] * r)
    return q


def quadratic_roots(coeffs):
    # вычисление всех корней квадратного уравнения a*lambda^2 + b*lambda + c = 0
    a, b, c = coeffs
    d = b * b - 4 * a * c
    if d >= 0:
        return [(-b - d ** 0.5) / (2 * a), (-b + d ** 0.5) / (2 * a)]
    re = -b / (2 * a)
    im = (-d) ** 0.5 / (2 * a)
    return [complex(re, im), complex(re, -im)]


def eigenvalues(alpha, n, q):
    # СЗ уравнения det(lambda*E - alpha) = 0
    M = [[[-alpha[i][i], 1.0] if i == j else [-alpha[i][j]] for j in range(n)] for i in range(n)]
    coeffs = det_poly(M)[::-1]

    roots = find_real_roots(coeffs, q + 1)
    rest = coeffs
    for r in roots:
        rest = deflate(rest, r)

    if len(rest) == 3:
        roots += quadratic_roots(rest)
    elif len(rest) == 2:
        roots.append(-rest[1] / rest[0])
    return roots


def check_convergence(A, alpha, n, q):
    # Диагональное преобладаение матрицы
    dom = False
    for i in range(n):
        s = sum(abs(A[i][j]) for j in range(n) if j != i)
        ok = abs(A[i][i]) > s
        # print(f"   |a_{i + 1}{i + 1}| = {abs(A[i][i])} {'>' if ok else '<='} sum|a_ij| = {s}")
        if not ok:
            break
    else:
        dom = True

    # Необходимо и достаточное условие сходимости
    roots = eigenvalues(alpha, n, q)
    # print("det(lambda*E - alpha) = 0:")
    # print("СЗ: " + ", ".join(f"({r.real:.6f} {r.imag:+.6f}i)" for r in roots))

    radius = max(abs(r) for r in roots)
    # print(f"Спектральный радиус = {radius:.6f} -> ", end="")ß

    conditions = {
        "Достаточное условие Якоби": dom,
        "Необходимое и достаточное условие": radius < 1,
    }
    return conditions


def simple_iterations(A, b, n, eps):
    print("\nМетод простых итераций (Якоби)")
    beta = [[b[i][0] / A[i][i]] for i in range(n)]
    alpha = [[-A[i][j] / A[i][i] if i != j else 0 for j in range(n)] for i in range(n)]
    q = norm_mat(alpha)

    conditions = check_convergence(A, alpha, n, q)
    print("Проверка условий сходимости:")
    for name, value in conditions.items():
        print(f"{name} = {value}")

    # print(f"beta:\n{format_matrix(beta)}")
    # print(f"alpha:\n{format_matrix(alpha)}")

    x = copy(beta)
    iterations = 0

    while True:
        x_prev = copy(x)
        x = add(beta, mul(alpha, x_prev))
        iterations += 1

        delta = norm_vec(subtract(x, x_prev))

        # print(f"\nИтерация {k + 1}:")
        # print(f"  x^{k}   = {format_vector(x_prev)}")
        # print(f"  x^{k + 1} = beta + alpha * x^{k} = {format_vector(x)}")
        # print(f"  ||x^{k + 1} - x^{k}|| = {delta:.6f}")
        if q < 1:
            eps_k = q / (1 - q) * delta
            # print(f"  eps^{k + 1} = ||alpha||/(1 - ||alpha||) * ||x^{k + 1} - x^{k}|| = {q:.6f}/{1 - q:.6f} * {delta:.6f} = {eps_k:.6f}")
        else:
            eps_k = delta
        # print(f"  eps^{k + 1} = {eps_k:.6f} {'<' if eps_k < eps else '>='} eps = {eps}")

        if eps_k < eps:
            break

    print(f"количество итераций = {iterations}")
    return x


def seidel(A, b, n, eps):
    print("\nМетод Зейделя")
    beta = [[b[i][0] / A[i][i]] for i in range(n)]
    alpha = [[-A[i][j] / A[i][i] if i != j else 0 for j in range(n)] for i in range(n)]
    q = norm_mat(alpha)

    conditions = check_convergence(A, alpha, n, q)
    print("Проверка условий сходимости:")
    for name, value in conditions.items():
        print(f"{name} = {value}")

    x = copy(beta)
    iterations = 0

    while True:
        x_prev = copy(x)
        for i in range(n):
            summ = beta[i][0] + sum(alpha[i][j] * x[j][0] for j in range(i)) + sum(alpha[i][j] * x_prev[j][0] for j in range(i, n))
            x[i][0] = summ
        iterations += 1
        delta = norm_vec(subtract(x, x_prev))
        if q < 1:
            C = [[alpha[i][j] if j > i else 0.0 for j in range(n)] for i in range(n)]
            eps_k = norm_mat(C) / (1 - q) * delta
        else:
            eps_k = delta
        if eps_k < eps:
            break

    print(f"количество итераций = {iterations}")
    return x


def main():
    n = 4
    eps = 0.001
    A = [[20, 5, 7, 1], [-1, 13, 0, -7],
         [4, -6, 17, 5], [-9, 8, 4, -25]]
    b = [[-117], [-1], [49], [-21]]

    print(f"Размерность A: {len(A)}x{len(A[0])}")
    print(f"Размерность b: {len(b)}")
    print(f"Матрицу можно умножить на вектор: {len(A[0]) == len(b)}")

    x = simple_iterations(A, b, n, eps)
    x_seidel = seidel(A, b, n, eps)

    print("\nЗначения x методом простых итераций:\n", format_matrix(x))
    print("Значения x методом Зейделя:\n", format_matrix(x_seidel))

    x_np = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))
    print("\nРешение через numpy:\n", x_np)


if __name__ == "__main__":
    main()
