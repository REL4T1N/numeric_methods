from dataclasses import dataclass
import math

import matplotlib.pyplot as plt


MAX_ITERATIONS = 10_000
REGION = ((0.4, 0.5), (2.1, 2.2))


@dataclass
class IterationResult:
    root: tuple[float, float]
    iterations: int
    approximations: list[tuple[float, float]]
    errors: list[float]
    error_estimates: list[float]


def functions(x):
    x1, x2 = x
    if x1 <= -1:
        raise ValueError("lg(x1+1) не определён при x1 <= -1")
    return x1 - math.cos(x2) - 1, x2 - math.log10(x1 + 1) - 2


def jacobian(x):
    x1, x2 = x
    return ((1.0, math.sin(x2)),
            (-1.0 / ((x1 + 1) * math.log(10)), 1.0))


def phi(x):
    x1, x2 = x
    return 1 + math.cos(x2), 2 + math.log10(x1 + 1)


def phi_jacobian(x):
    x1, x2 = x
    return ((0.0, -math.sin(x2)),
            (1.0 / ((x1 + 1) * math.log(10)), 0.0))


def norm_inf(vector):
    return max(abs(value) for value in vector)


def matrix_norm_inf(matrix):
    return max(sum(abs(value) for value in row) for row in matrix)


def region_center(region):
    (x1_min, x1_max), (x2_min, x2_max) = region
    return ((x1_min + x1_max) / 2,
            (x2_min + x2_max) / 2)


def check_simple_iteration_conditions(region=REGION):
    """проверяет phi(G) подмножество G и max_G ||phi'(x)|| = q < 1."""
    (x1_min, x1_max), (x2_min, x2_max) = region

    if x1_min <= -1:
        raise ValueError("lg(x1 + 1) не определён при x1 <= -1")

    # На G функция phi_1 убывает по x_2, а phi_2 возрастает по x_1.
    phi_min = phi((x1_min, x2_max))
    phi_max = phi((x1_max, x2_min))
    maps_into_itself = (
        x1_min <= phi_min[0] <= phi_max[0] <= x1_max
        and x2_min <= phi_min[1] <= phi_max[1] <= x2_max
    )
    if not maps_into_itself:
        raise ValueError(
            f"phi(G) не является подмножеством G: min={phi_min}, max={phi_max}"
        )

    # На G обе суммы модулей по строкам достигают максимумов
    # при x_1=x1_min и x_2=x2_min.
    q = matrix_norm_inf(phi_jacobian((x1_min, x2_min)))
    if q >= 1:
        raise ValueError(f"Условие сжатия не выполнено: q={q}")
    return q


def calculate_newton_correction(x):
    """Решает J(x) * dx = F(x) по правилу Крамера."""
    f1, f2 = functions(x)
    (a, b), (c, d) = jacobian(x)
    determinant = a * d - b * c
    if abs(determinant) < 1e-15:
        raise ZeroDivisionError("Матрица Якоби вырождена")

    dx1 = (d * f1 - b * f2) / determinant
    dx2 = (-c * f1 + a * f2) / determinant
    return dx1, dx2


def _reference_solution():
    """решение x для графика погрешности"""
    x = (0.45, 2.16)
    for _ in range(30):
        dx1, dx2 = calculate_newton_correction(x)
        x = x[0] - dx1, x[1] - dx2
        if norm_inf((dx1, dx2)) < 1e-15:
            break
    return x


def simple_iteration(eps, region=REGION):
    q = check_simple_iteration_conditions(region)
    (x1_min, x1_max), (x2_min, x2_max) = region
    
    x = region_center(region)
    reference = _reference_solution()

    approximations = [x]
    errors = [norm_inf((x[0] - reference[0], x[1] - reference[1]))]
    estimates = []

    for _ in range(MAX_ITERATIONS):
        next_x = phi(x)
        if not (x1_min <= next_x[0] <= x1_max and x2_min <= next_x[1] <= x2_max):
            raise RuntimeError("Приближение вышло из области сходимости G")
        
        step = norm_inf((next_x[0] - x[0], next_x[1] - x[1]))
        estimate = q / (1 - q) * step

        approximations.append(next_x)
        errors.append(norm_inf((next_x[0] - reference[0], next_x[1] - reference[1])))
        estimates.append(estimate)

        x = next_x
        if estimate <= eps:
            return IterationResult(x, len(estimates), approximations, errors, estimates)
    raise RuntimeError("Метод простой итерации превысил лимит итераций")


def newton(eps, region=REGION):
    reference = _reference_solution()
    x = region_center(region)

    approximations = [x]
    errors = [norm_inf((x[0] - reference[0], x[1] - reference[1]))]
    steps = []

    for _ in range(MAX_ITERATIONS):
        dx1, dx2 = calculate_newton_correction(x)
        next_x = x[0] - dx1, x[1] - dx2
        step = norm_inf((next_x[0] - x[0], next_x[1] - x[1]))

        approximations.append(next_x)
        errors.append(norm_inf((next_x[0] - reference[0], next_x[1] - reference[1])))
        steps.append(step)

        x = next_x
        if step <= eps:
            return IterationResult(x, len(steps), approximations, errors, steps)
        
    raise RuntimeError("Метод Ньютона превысил лимит итераций")


def plot_system(filename="lab2-2-system.png", show=False):
    """Графическая локализация: x1=1+cos(x2) и x2=2+lg(x1+1)."""
    x2_values = [1.9 + i * 0.002 for i in range(251)]
    curve1_x1 = [1 + math.cos(x2) for x2 in x2_values]
    x1_values = [0.05 + i * 0.0025 for i in range(301)]
    curve2_x2 = [2 + math.log10(x1 + 1) for x1 in x1_values]
    plt.figure(figsize=(7, 5))
    plt.plot(curve1_x1, x2_values, label=r"$x_1=1+\cos x_2$")
    plt.plot(x1_values, curve2_x2, label=r"$x_2=2+\lg(x_1+1)$")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_errors(results, filename="lab2-2-errors.png", show=False):
    plt.figure(figsize=(10.5, 4.5))
    for label, result in results.items():
        safe_errors = [max(error, 1e-16) for error in result.errors]
        plt.semilogy(range(len(safe_errors)), safe_errors, marker="o", label=label)
    max_iteration = max(len(result.errors) - 1 for result in results.values())
    plt.xticks(range(max_iteration + 1))
    plt.xlabel("Итерации")
    plt.ylabel("Погрешность")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    raw_eps = input("Введите точность epsilon [1e-6]: ").strip()
    eps = float(raw_eps) if raw_eps else 1e-6

    iteration_result = simple_iteration(eps)
    newton_result = newton(eps)
    
    results = {"Метод простых итераций": iteration_result,
               "Метод Ньютона": newton_result}
    
    for name, result in results.items():
        residual = norm_inf(functions(result.root))
        print(f"{name}: x1={result.root[0]:.10f}, x2={result.root[1]:.10f}, "
              f"итераций={result.iterations}, невязка={residual:.3e}")
    
    plot_system()
    plot_errors(results)
    
    print("\nГрафики сохранены в lab2-2-system.png и lab2-2-errors.png")


if __name__ == "__main__":
    main()
