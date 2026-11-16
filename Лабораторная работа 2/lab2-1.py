from dataclasses import dataclass
import math

import matplotlib.pyplot as plt


MAX_ITERATIONS = 10_000
ROOT_INTERVALS = ((1.0, 1.5), (4.0, 4.5))


@dataclass
class IterationResult:
    root: float
    iterations: int
    approximations: list[float]
    errors: list[float]
    error_estimates: list[float]


def f(x):
    return math.cos(x) + 0.25 * x - 0.5


def df(x):
    return -math.sin(x) + 0.25


def d2f(x):
    return -math.cos(x)


def phi(x, lambda_value):
    """Эквивалентное уравнение phi(x) = x - lambda*f(x)."""
    return x - lambda_value * f(x)


def dphi(x, lambda_value):
    return 1 - lambda_value * df(x)


def derivative_critical_points(a, b):
    """Точки внутри [a,b], где f''(x)=0, то есть экстремумы f'(x)."""
    points = []
    k_min = math.ceil((a - math.pi / 2) / math.pi)
    k_max = math.floor((b - math.pi / 2) / math.pi)
    for k in range(k_min, k_max + 1):
        x = math.pi / 2 + math.pi * k
        if a < x < b:
            points.append(x)
    return points


def derivative_bounds(a, b):
    """Точные минимум и максимум f' по концам и точкам f''=0."""
    points = [a, *derivative_critical_points(a, b), b]
    values = [df(x) for x in points]
    return min(values), max(values)


def second_derivative_bounds(a, b):
    """Точные границы f''; её экстремумы удовлетворяют f'''(x)=sin(x)=0."""
    points = [a, b]
    for k in range(math.ceil(a / math.pi), math.floor(b / math.pi) + 1):
        x = math.pi * k
        if a < x < b:
            points.append(x)
    values = [d2f(x) for x in points]
    return min(values), max(values)


def calculate_iteration_parameters(a, b):
    """Вычисляет минимаксное lambda и коэффициент сжатия q.

    При m <= f'(x) <= M и постоянном знаке f' выбор lambda=2/(m+M)
    делает крайние значения phi'(x)=1-lambda*f(x)' равными по модулю и
    противоположными по знаку. Такой выбор минимизирует max|phi'|.
    """
    derivative_min, derivative_max = derivative_bounds(a, b)
    if derivative_min <= 0 <= derivative_max: # Если f'(x) меняет знак на [a,b], то существует c из [a,b]:
                                              # f'(c)=0 => phi'(c)=1-lambda*f'(c)=1.
                                              # Тогда |phi'(c)|=1<=q, что нарушает условие q<1.
        raise ValueError("f'(x) меняет знак или обращается в ноль на [a,b]")
    
    lambda_value = 2 / (derivative_min + derivative_max)
    q = max(abs(1 - lambda_value * derivative_min),
            abs(1 - lambda_value * derivative_max))
    
    return lambda_value, q


def phi_critical_points(a, b, lambda_value):
    """Точки внутри [a,b], где phi'(x)=0."""
    sine_value = 0.25 - 1 / lambda_value
    if not abs(sine_value) <= 1:
        return []
    
    alpha = math.asin(sine_value)
    points = []
    for base in (alpha, math.pi - alpha):
        k_min = math.ceil((a - base) / (2 * math.pi))
        k_max = math.floor((b - base) / (2 * math.pi))
        for k in range(k_min, k_max + 1):
            x = base + 2 * math.pi * k
            if a < x < b:
                points.append(x)
    return sorted(set(points))


def _reference_root(a, b, tolerance=1e-14):
    """Опорный корень для графика погрешности."""
    left, right = a, b
    while right - left > 2 * tolerance:
        middle = (left + right) / 2
        if f(left) * f(middle) <= 0:
            right = middle
        else:
            left = middle
    return (left + right) / 2


def check_root_isolation(a, b):
    """Проверяет теорему 2.1: смену знака и строгую монотонность."""
    sign_change = f(a) * f(b) < 0

    derivative_min, derivative_max = derivative_bounds(a, b)
    monotone = derivative_min > 0 or derivative_max < 0

    if not sign_change or not monotone:
        raise ValueError(f"Отрезок [{a}, {b}] не изолирует корень")
    
    return True


def check_simple_iteration_conditions(a, b):
    """Аналитически проверяет phi([a,b]) подмножество [a,b] и max|phi'|<1."""
    lambda_value, q = calculate_iteration_parameters(a, b)

    points = [a, *phi_critical_points(a, b, lambda_value), b]
    values = [phi(x, lambda_value) for x in points]

    phi_min, phi_max = min(values), max(values)
    maps_into_itself = phi_min >= a and phi_max <= b
    if not maps_into_itself:
        raise ValueError(
            f"phi([a,b])=[{phi_min}; {phi_max}] не является подмножеством [a,b]=[{a}; {b}]"
        )
    
    if q >= 1:
        raise ValueError(f"Условие сжатия не выполнено: q={q} >= 1")
    
    return lambda_value, q


def simple_iteration(a, b, eps):
    """Метод простой итерации с критерием выхода q/(1-q)|dx|."""
    check_root_isolation(a, b)
    lambda_value, q = check_simple_iteration_conditions(a, b)

    reference = _reference_root(a, b)
    x = (a + b) / 2
    approximations, errors, estimates = [x], [abs(x - reference)], []

    for _ in range(MAX_ITERATIONS):
        next_x = phi(x, lambda_value)
        estimate = q / (1 - q) * abs(next_x - x)
        approximations.append(next_x)
        errors.append(abs(next_x - reference))
        estimates.append(estimate)
        x = next_x
        if estimate <= eps:
            return IterationResult(x, len(estimates), approximations, errors, estimates)
        
    raise RuntimeError("Метод простой итерации превысил лимит итераций")


def choose_newton_initial(a, b):
    """Выбирает конец отрезка, для которого f(x0)f''(x0)>0 (теорема 2.2)."""
    first_min, first_max = derivative_bounds(a, b)
    second_min, second_max = second_derivative_bounds(a, b)

    constant_sign = ((first_min > 0 or first_max < 0) and (second_min > 0 or second_max < 0))
    if not constant_sign:
        raise ValueError("f' и f'' должны сохранять знаки на [a,b]")
    
    for x0 in (a, b):
        if f(x0) * d2f(x0) > 0:
            return x0
        
    raise ValueError("Ни один конец отрезка не удовлетворяет f(x0)f''(x0)>0")


def newton(a, b, eps):
    check_root_isolation(a, b)
    x = choose_newton_initial(a, b)

    reference = _reference_root(a, b)
    approximations, errors, steps = [x], [abs(x - reference)], []

    for _ in range(MAX_ITERATIONS):
        derivative = df(x)
        if abs(derivative) < 1e-15:
            raise ZeroDivisionError("В методе Ньютона получено f'(x)=0")
        
        next_x = x - f(x) / derivative
        step = abs(next_x - x)

        approximations.append(next_x)
        errors.append(abs(next_x - reference))
        steps.append(step)

        x = next_x
        if step <= eps:
            return IterationResult(x, len(steps), approximations, errors, steps)
        
    raise RuntimeError("Метод Ньютона превысил лимит итераций")


def plot_function(filename="lab2-1-function.png", show=False):
    xs = [i * 0.01 for i in range(601)]
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, [f(x) for x in xs], label=r"$f(x)=\cos x+x/4-1/2$")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_errors(results, filename="lab2-1-errors.png", show=False):
    """Строит график фактической погрешности по итерациям в логарифмическом масштабе."""
    plt.figure(figsize=(10.5, 4.5))
    for label, result in results.items():
        iterations = range(len(result.errors))
        safe_errors = [max(error, 1e-16) for error in result.errors]
        plt.semilogy(iterations, safe_errors, marker="o", label=label)
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
    results = {}

    for number, (a, b) in enumerate(ROOT_INTERVALS, 1):
        iteration_result = simple_iteration(a, b, eps)
        newton_result = newton(a, b, eps)
        results[f"Метод простых итераций, корень {number}"] = iteration_result
        results[f"Метод Ньютона, корень {number}"] = newton_result

        print(f"  Метод простой итерации: x={iteration_result.root:.10f}, "
              f"итераций={iteration_result.iterations}, |f(x)|={abs(f(iteration_result.root)):.3e}")
        print(f"  Метод Ньютона: x={newton_result.root:.10f}, "
              f"итераций={newton_result.iterations}, |f(x)|={abs(f(newton_result.root)):.3e}")
        
    plot_function()
    plot_errors(results)
    
    print("\nГрафики сохранены в lab2-1-function.png и lab2-1-errors.png")


if __name__ == "__main__":
    main()
