import math
import matplotlib.pyplot as plt
from lab3_common import HERE, check_nodes, plot_errors

XS = [0.2, 0.6, 1.0, 1.4]
TARGET = 0.8


def lagrange(xs, ys, x):
    check_nodes(xs, ys)
    return sum(ys[i] * math.prod((x-xs[j])/(xs[i]-xs[j])
                 for j in range(len(xs)) if j != i) for i in range(len(xs)))


def divided_differences(xs, ys):
    check_nodes(xs, ys)
    table = [list(ys)]
    for order in range(1, len(xs)):
        previous = table[-1]
        table.append([(previous[i+1]-previous[i])/(xs[i+order]-xs[i])
                      for i in range(len(xs)-order)])
    return table


def newton(xs, ys, x):
    coefficients = [column[0] for column in divided_differences(xs, ys)]
    result = coefficients[-1]
    for i in range(len(coefficients)-2, -1, -1):
        result = coefficients[i] + (x-xs[i])*result
    return result


def plot_interpolation(xs, ys, filename="lab3-1-interpolation.png"):
    check_nodes(xs, ys)
    grid = [xs[0] + (xs[-1] - xs[0]) * i / 500 for i in range(501)]
    exact = [math.log(x) for x in grid]
    interpolated = [newton(xs, ys, x) for x in grid]
    node_errors = [abs(newton(xs, ys, x) - y) for x, y in zip(xs, ys)]

    figure, (function_axis, error_axis) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    function_axis.plot(grid, exact, label="ln(x)")
    function_axis.plot(grid, interpolated, "--", label="P₃(x)")
    function_axis.scatter(xs, ys, color="black", zorder=3, label="Заданные узлы")
    function_axis.set_ylabel("y")
    function_axis.grid(True)
    function_axis.legend()

    error_axis.plot(grid, [p - f for p, f in zip(interpolated, exact)],
                    label="P₃(x) − ln(x)")
    error_axis.axhline(0, color="gray", linewidth=0.8)
    error_axis.scatter(xs, [0.0] * len(xs), color="black", zorder=3,
                       label="Ошибка в узлах = 0")
    error_axis.set_xlabel("x")
    error_axis.set_ylabel("Ошибка")
    error_axis.set_xticks(xs)
    error_axis.grid(True)
    error_axis.legend()
    figure.tight_layout()
    figure.savefig(HERE / filename, dpi=150)
    plt.close(figure)
    return max(node_errors)


def main():
    ys = [math.log(x) for x in XS]
    exact = math.log(TARGET)
    print("Варианты а) и б): заданные списки узлов совпадают.")
    print("Разделённые разности:", [row[0] for row in divided_differences(XS, ys)])

    for name, method in (("Лагранж", lagrange), ("Ньютон", newton)):
        value = method(XS, ys, TARGET)
        print(f"{name}: P({TARGET})={value:.12f}; |ln({TARGET})-P|={abs(exact-value):.12g}")

    bound = 6/XS[0]**4/math.factorial(4)*abs(math.prod(TARGET-x for x in XS))
    print(f"Априорная верхняя граница по остатку: {bound:.12g}")

    counts = [2, 3, 4]
    errors = {name: [abs(method(XS[:n], ys[:n], TARGET)-exact) for n in counts]
              for name, method in (("Лагранж", lagrange), ("Ньютон", newton))}
    plot_errors(counts, errors, "lab3-1-errors.png")

    maximum_node_error = plot_interpolation(XS, ys)
    print(f"Максимальное |P₃(xᵢ) − yᵢ| в узлах: {maximum_node_error:.3e}")

if __name__ == "__main__":
    main()
