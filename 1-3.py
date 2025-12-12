import math
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import poisson as pois, kstwobign, chi2
from scipy.stats import norm
import statistics

def poisson(theta):
    k = 0
    p = 1.0
    L = math.exp(-theta)
    while p > L:
        k += 1
        U = random.uniform(0, 1)
        p *= U
    return k - 1


def normal(theta, sigma):
    U1 = random.random()
    U2 = random.random()
    Z1 = math.sqrt(-2 * math.log(U1)) * math.cos(2 * math.pi * U2)
    return theta + sigma * Z1


theta_poisson = 17.5
theta_normal = 6.0
sigma_normal = 15.5

# ПОСТРОЕНИЕ ДЛЯ ПУАССОНА

# Генерация выборки
sample_size = 1000
sample0 = [poisson(theta_poisson) for _ in range(sample_size)]

# Создание графика
plt.figure(figsize=(12, 6))

# Гистограмма экспериментальной выборки
plt.hist(sample0, bins=range(0, max(sample0) + 1), density=True,
         alpha=0.7, color='lightblue', edgecolor='black',
         label='Экспериментальная выборка')

# Теоретическое распределение Пуассона
x = np.arange(0, max(sample0) + 1)
theoretical_probs = pois.pmf(x, theta_poisson)

plt.plot(x, theoretical_probs, 'ro-', linewidth=2, markersize=6,
         label='Теоретическое распределение Пуассона')

# Настройки графика
plt.xlabel('Значение k')
plt.ylabel('Вероятность P(X=k)')
plt.title(f'Сравнение экспериментальной выборки с теоретическим распределением Пуассона (theta={theta_poisson})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, max(sample0))

# Добавляем теоретические вероятности на график
for i, prob in enumerate(theoretical_probs):
    if prob > 0.01:  # показываем только вероятности > 1%
        plt.annotate(f'{prob:.3f}', (x[i], prob),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=8)

plt.tight_layout()
plt.show()

# ПОСТРОЕНИЕ ДЛЯ НОРМАЛЬНОГО

sample1 = [normal(theta_normal, sigma_normal) for _ in range(sample_size)]
# Создание графика
plt.figure(figsize=(12, 6))

# Гистограмма экспериментальной выборки
plt.hist(sample1, bins=50, density=True,
         alpha=0.7, color='lightblue', edgecolor='black',
         label='Экспериментальная выборка')

# Теоретическая плотность нормального распределения
x = np.linspace(min(sample1), max(sample1), 1000)
theoretical_pdf = norm.pdf(x, theta_normal, sigma_normal)

plt.plot(x, theoretical_pdf, 'r-', linewidth=2,
         label='Теоретическая плотность распределения')

# Настройки графика
plt.xlabel('Значение x')
plt.ylabel('Плотность вероятности f(x)')
plt.title(
    f'Сравнение экспериментальной выборки с теоретическим нормальным распределением\n(theta={theta_normal}, sigma={sigma_normal})')
plt.legend()
plt.grid(True, alpha=0.3)

# Добавляем вертикальные линии для характеристик
plt.axvline(theta_normal, color='red', linestyle='--', alpha=0.7, label=f'theta = {theta_normal}')
plt.axvline(theta_normal + sigma_normal, color='orange', linestyle='--', alpha=0.7,
            label=f'theta + sigma = {theta_normal + sigma_normal:.1f}')
plt.axvline(theta_normal - sigma_normal, color='orange', linestyle='--', alpha=0.7,
            label=f'theta - sigma = {theta_normal - sigma_normal:.1f}')

plt.legend()
plt.tight_layout()
plt.show()

sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
num_samples_per_size = 5


def generate_samples(distribution, params, sizes, num_repeats):
    samples = {}
    for size in sizes:
        samples[size] = []
        for _ in range(num_repeats):
            if distribution == poisson:
                sample = [poisson(*params) for _ in range(size)]
            else:
                sample = [normal(*params) for _ in range(size)]
            samples[size].append(sample)
    return samples


poisson_samples = generate_samples(poisson, [theta_poisson], sample_sizes, num_samples_per_size)
normal_samples = generate_samples(normal, [theta_normal, sigma_normal], sample_sizes, num_samples_per_size)


def empirical_cdf(sample, t):
    count = sum(1 for x in sample if x < t)
    return count / len(sample)


# Теоретические функции распределения
def theoretical_poisson_cdf(t):
    if t < 0:
        return 0
    cdf = 0.0
    for k in range(0, int(t) + 1):
        cdf += math.exp(-theta_poisson) * (theta_poisson ** k) / math.factorial(k)
    return cdf


def theoretical_normal_cdf(t):
    return 0.5 * (1 + math.erf((t - theta_normal) / (sigma_normal * math.sqrt(2))))


def plot_empirical_cdfs(samples, theoretical_cdf, title, x_range=None):
    plt.figure(figsize=(12, 8))

    # Цвета для разных размеров выборок
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_sizes)))

    # Построение теоретической функции распределения
    if x_range is None:
        # Определяем диапазон на основе всех выборок
        all_values = []
        for size_samples in samples.values():
            for sample in size_samples:
                all_values.extend(sample)
        x_min, x_max = min(all_values), max(all_values)
        x_range = np.linspace(x_min - 1, x_max + 1, 1000)

    theoretical_values = [theoretical_cdf(x) for x in x_range]
    plt.plot(x_range, theoretical_values, 'k-', linewidth=3, label='Теоретическая ФР', alpha=0.8)

    # Построение эмпирических функций распределения для каждого размера выборки
    for i, size in enumerate(sample_sizes):
        color = colors[i]
        for j, sample in enumerate(samples[size]):
            # Сортируем выборку для построения эмпирической ФР
            sorted_sample = sorted(sample)
            n = len(sorted_sample)

            # Подготавливаем данные для ступенчатой функции
            x_vals = []
            y_vals = []

            # Начинаем с -∞
            x_vals.append(sorted_sample[0] - 1)
            y_vals.append(0)

            # Добавляем ступеньки в каждой точке выборки
            for k in range(n):
                x_vals.append(sorted_sample[k])
                y_vals.append(k / n)
                if k < n - 1:
                    x_vals.append(sorted_sample[k])
                    y_vals.append((k + 1) / n)

            # Заканчиваем на +∞
            x_vals.append(sorted_sample[-1] + 1)
            y_vals.append(1)

            plt.step(x_vals, y_vals, where='post', alpha=0.7,
                     color=color, label=f'n={size}' if j == 0 else "")

    plt.xlabel('t')
    plt.ylabel('F(t)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_separate_empirical_cdfs(samples, theoretical_cdf, distribution_name):
    # Определяем общий диапазон x для всех графиков
    all_values = []
    for size_samples in samples.values():
        for sample in size_samples:
            all_values.extend(sample)
    x_min, x_max = min(all_values), max(all_values)
    x_range = np.linspace(x_min - 1, x_max + 1, 1000)

    # Строим отдельный график для каждого размера выборки
    for size in sample_sizes:
        plt.figure(figsize=(10, 6))

        # Теоретическая функция распределения
        theoretical_values = [theoretical_cdf(x) for x in x_range]
        plt.plot(x_range, theoretical_values, 'r-', linewidth=2,
                 label=f'Теоретическая ФР', alpha=0.8)

        # Эмпирические функции распределения для текущего размера
        colors = plt.cm.viridis(np.linspace(0, 1, len(samples[size])))

        for i, sample in enumerate(samples[size]):
            # Сортируем выборку для построения эмпирической ФР
            sorted_sample = sorted(sample)
            n = len(sorted_sample)

            # Подготавливаем данные для ступенчатой функции
            x_vals = []
            y_vals = []

            # Начинаем с -inf
            x_vals.append(sorted_sample[0] - 1)
            y_vals.append(0)

            # Добавляем ступеньки в каждой точке выборки
            for k in range(n):
                x_vals.append(sorted_sample[k])
                y_vals.append(k / n)
                if k < n - 1:
                    x_vals.append(sorted_sample[k])
                    y_vals.append((k + 1) / n)

            # Заканчиваем на +inf
            x_vals.append(sorted_sample[-1] + 1)
            y_vals.append(1)

            plt.step(x_vals, y_vals, where='post', alpha=0.7, color=colors[i],
                     label=f'Эмпирическая ФР (шаг {i + 1})')

        plt.xlabel('t')
        plt.ylabel('F(t)')
        plt.title(f'{distribution_name} \nРазмер выборки n={size}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


plot_separate_empirical_cdfs(poisson_samples, theoretical_poisson_cdf, "Распределение Пуассона")
plot_separate_empirical_cdfs(normal_samples, theoretical_normal_cdf, "Нормальное распределение")

plot_empirical_cdfs(poisson_samples, theoretical_poisson_cdf, "Эмпирические функции распределения (Пуассон)")
plot_empirical_cdfs(normal_samples, theoretical_normal_cdf, "Эмпирические функции распределения (Нормальное)")


def compute_supremum_difference(sample_1, sample_2):
    all_points = sorted(set(sample_1 + sample_2))

    max_diff = 0
    for point in all_points:
        fn = empirical_cdf(sample_1, point)
        fm = empirical_cdf(sample_2, point)
        diff = abs(fn - fm)
        if diff > max_diff:
            max_diff = diff

    return max_diff


def compute_D_statistics(samples):
    D_results = defaultdict(dict)

    sizes = list(samples.keys())

    for i, n in enumerate(sizes):
        for j, m in enumerate(sizes):
            if n <= m:
                D_values = []
                for sample_n in samples[n]:
                    for sample_m in samples[m]:
                        sup_diff = compute_supremum_difference(sample_n, sample_m)
                        D = math.sqrt((n * m) / (n + m)) * sup_diff
                        D_values.append(D)

                D_results[(n, m)] = {
                    'mean': np.mean(D_values),
                    'std': np.std(D_values),
                    'min': np.min(D_values),
                    'max': np.max(D_values),
                    'all_values': D_values
                }

    return D_results


D_poisson = compute_D_statistics(poisson_samples)
D_normal = compute_D_statistics(normal_samples)


# Вывод результатов
def print_D_statistics(D_results):
    sorted_pairs = sorted(D_results.keys(), key=lambda x: (x[0], x[1]))

    print(
        f"\n{'Пара (n,m)':<12} {'Среднее':<10} {'Станд. откл.':<12} {'Минимум':<10} {'Максимум':<10}")
    print(f"{'-' * 70}")

    for pair in sorted_pairs:
        stats = D_results[pair]
        n, m = pair

        mean_val = f"{stats['mean']:.4f}"
        std_val = f"{stats['std']:.4f}"
        min_val = f"{stats['min']:.4f}"
        max_val = f"{stats['max']:.4f}"

        print(f"({n},{m})   {mean_val:<10} {std_val:<12} {min_val:<10} {max_val:<10}")


print_D_statistics(D_poisson)
print_D_statistics(D_normal)


# Визуализация статистики D
def plot_D_statistics(D_results, title):
    sizes = sample_sizes
    n_sizes = len(sizes)

    # Создаем матрицу для тепловой карты
    heatmap_data = np.zeros((n_sizes, n_sizes))

    for i, n in enumerate(sizes):
        for j, m in enumerate(sizes):
            if n <= m:
                heatmap_data[i, j] = D_results[(n, m)]['mean']
                heatmap_data[j, i] = D_results[(n, m)]['mean']  # Симметрично

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Среднее значение D_{m,n}')
    plt.xticks(range(n_sizes), sizes)
    plt.yticks(range(n_sizes), sizes)
    plt.xlabel('m (размер выборки)')
    plt.ylabel('n (размер выборки)')
    plt.title(title)

    # Добавляем аннотации
    for i in range(n_sizes):
        for j in range(n_sizes):
            plt.text(j, i, f'{heatmap_data[i, j]:.3f}',
                     ha='center', va='center', color='white' if heatmap_data[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.show()


plot_D_statistics(D_poisson, "Статистика D_{m,n} Распределение Пуассона")
plot_D_statistics(D_normal, "Статистика D_{m,n} Нормальное распределение")

def plot_histograms_and_polygons(samples, theoretical_pmf_or_pdf, title, distribution_type, sample_size_to_show=None):
    if sample_size_to_show is None:
        sizes_to_plot = [5, 10, 100, 200, 400, 600, 800, 1000]
    else:
        sizes_to_plot = [sample_size_to_show]

    for size in sizes_to_plot:
        # Берем первую выборку данного размера для примера
        sample = samples[size][0]

        plt.figure(figsize=(12, 8))

        if distribution_type == 'discrete':
            # Для дискретного распределения (Пуассон)
            unique_values, counts = np.unique(sample, return_counts=True)
            probabilities = counts / len(sample)

            # Полигон частот
            plt.plot(unique_values, probabilities, 'bo-', linewidth=2, markersize=6, label='Полигон частот')

            # Теоретическая функция вероятности
            x_min, x_max = min(unique_values), max(unique_values)
            x_theoretical = np.arange(max(0, x_min - 1), x_max + 2)
            y_theoretical = [theoretical_pmf_or_pdf(x) for x in x_theoretical]
            plt.plot(x_theoretical, y_theoretical, 'r-', linewidth=2, alpha=0.7, label='Теоретическая ФВ')

            # Гистограмма (столбчатая диаграмма)
            plt.bar(unique_values, probabilities, alpha=0.3, color='blue', label='Гистограмма частот')

        else:
            # Для непрерывного распределения (Нормальное)
            # Гистограмма
            n, bins, patches = plt.hist(sample, bins='auto', density=True, alpha=0.7,
                                        color='lightblue', edgecolor='black', label='Гистограмма')

            # Полигон частот
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            plt.plot(bin_centers, n, 'bo-', linewidth=2, markersize=4, label='Полигон частот')

            # Теоретическая плотность распределения
            x_theoretical = np.linspace(min(sample), max(sample), 1000)
            y_theoretical = [theoretical_pmf_or_pdf(x) for x in x_theoretical]
            plt.plot(x_theoretical, y_theoretical, 'r-', linewidth=2, label='Теоретическая плотность')

        plt.xlabel('Значение')
        plt.ylabel('Вероятность / Плотность')
        plt.title(f'{title}\nРазмер выборки: {size}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# Теоретические функции для Пуассона и нормального распределения
def poisson_pmf(k):
    k = int(k)
    if k < 0:
        return 0
    return math.exp(-theta_poisson) * (theta_poisson ** k) / math.factorial(k)


def normal_pdf(x):
    return (1 / (sigma_normal * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - theta_normal) / sigma_normal) ** 2)


plot_histograms_and_polygons(poisson_samples, poisson_pmf, "Гистограмма и полигон частот (Пуассон)", 'discrete')
plot_histograms_and_polygons(normal_samples, normal_pdf, "Гистограмма и полигон частот (Нормальное)", 'continuous')

true_mean_poisson = theta_poisson
true_variance_poisson = theta_poisson

true_mean_normal = theta_normal
true_variance_normal = sigma_normal ** 2


def sample_mean(sample):
    return sum(sample) / len(sample)


def sample_variance(sample):
    n = len(sample)
    mean = sample_mean(sample)
    return sum((x - mean) ** 2 for x in sample) / n


def compute_sample_statistics(samples, true_mean, true_variance):
    statistics = {}

    for size, sample_list in samples.items():
        statistics[size] = {
            'sample_means': [],
            'sample_variances': [],
            'mean_errors': [],
            'variance_errors': []
        }

        for sample in sample_list:
            mean = sample_mean(sample)
            variance = sample_variance(sample)

            statistics[size]['sample_means'].append(mean)
            statistics[size]['sample_variances'].append(variance)
            statistics[size]['mean_errors'].append(abs(mean - true_mean))
            statistics[size]['variance_errors'].append(abs(variance - true_variance))

    return statistics


poisson_stats = compute_sample_statistics(poisson_samples, true_mean_poisson, true_variance_poisson)
normal_stats = compute_sample_statistics(normal_samples, true_mean_normal, true_variance_normal)


def print_sample_statistics(stats, distribution_name, true_mean, true_variance):
    print(f"\n{distribution_name}:")
    print(f"Истинное математическое ожидание: {true_mean:.4f}")
    print(f"Истинная дисперсия: {true_variance:.4f}")
    print("-" * 80)
    print(f"{'Размер':<8} {'Выб. среднее':<15} {'Ошибка ср.':<12} {'Выб. дисперсия':<15} {'Ошибка дисп.':<12}")
    print("-" * 80)

    for size in sample_sizes:
        means = stats[size]['sample_means']
        variances = stats[size]['sample_variances']
        mean_errors = stats[size]['mean_errors']
        variance_errors = stats[size]['variance_errors']

        # Усредняем по 5 выборкам
        avg_mean = np.mean(means)
        avg_variance = np.mean(variances)
        avg_mean_error = np.mean(mean_errors)
        avg_variance_error = np.mean(variance_errors)

        print(f"{size:<8} {avg_mean:<15.4f} {avg_mean_error:<12.4f} {avg_variance:<15.4f} {avg_variance_error:<12.4f}")


# Вывод резульатов
print_sample_statistics(poisson_stats, "Распределение Пуассона", true_mean_poisson, true_variance_poisson)
print_sample_statistics(normal_stats, "Нормальное распределение", true_mean_normal, true_variance_normal)


# Визуализация сходимости выборочных моментов
def plot_convergence(stats_poisson, stats_normal):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Для Пуассона
    sizes = sample_sizes
    poisson_mean_errors = [np.mean(stats_poisson[size]['mean_errors']) for size in sizes]
    poisson_variance_errors = [np.mean(stats_poisson[size]['variance_errors']) for size in sizes]

    # Для нормального распределения
    normal_mean_errors = [np.mean(stats_normal[size]['mean_errors']) for size in sizes]
    normal_variance_errors = [np.mean(stats_normal[size]['variance_errors']) for size in sizes]

    # Графики ошибок среднего
    axes[0, 0].plot(sizes, poisson_mean_errors, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Размер выборки')
    axes[0, 0].set_ylabel('Абсолютная ошибка')
    axes[0, 0].set_title('Сходимость выборочного среднего (Пуассон)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')

    axes[0, 1].plot(sizes, normal_mean_errors, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Размер выборки')
    axes[0, 1].set_ylabel('Абсолютная ошибка')
    axes[0, 1].set_title('Сходимость выборочного среднего (Нормальное)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')

    # Графики ошибок дисперсии
    axes[1, 0].plot(sizes, poisson_variance_errors, 'bo-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Размер выборки')
    axes[1, 0].set_ylabel('Абсолютная ошибка')
    axes[1, 0].set_title('Сходимость выборочной дисперсии (Пуассон)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')

    axes[1, 1].plot(sizes, normal_variance_errors, 'ro-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Размер выборки')
    axes[1, 1].set_ylabel('Абсолютная ошибка')
    axes[1, 1].set_title('Сходимость выборочной дисперсии (Нормальное)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')

    plt.tight_layout()
    plt.show()


plot_convergence(poisson_stats, normal_stats)

# Функция для метода моментов (Пуассон)
def method_of_moments_poisson(sample):
    return sum(sample) / len(sample)


# Вычисление оценок для распределения Пуассона
print("\nМЕТОД МОМЕНТОВ - РАСПРЕДЕЛЕНИЕ ПУАССОНА")
print("Истинный параметр θ =", theta_poisson)


for size in sample_sizes:
    print(f"\nРазмер выборки: {size}")
    print("-" * 40)
    for i, sample in enumerate(poisson_samples[size]):
        estimate = method_of_moments_poisson(sample)
        error = abs(estimate - theta_poisson)
        print(f"Выборка {i+1}: θ̂ = {estimate:.4f}, ошибка = {error:.4f}")

# Функция для метода моментов (нормальное распределение)
def method_of_moments_normal(sample):
    n = len(sample)
    a1 = sum(sample) / n
    a2 = sum(x ** 2 for x in sample) / n
    theta1 = a1
    theta2 = math.sqrt(a2 - a1 ** 2)
    return theta1, theta2


# Вычисление оценок для нормального распределения
print("\nМЕТОД МОМЕНТОВ - НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ")
print("Истинные параметры: theta_1 =", theta_normal, ", theta_2 =", sigma_normal)

for size in sample_sizes:
    print(f"\nРазмер выборки: {size}")
    print("-" * 60)
    for i, sample in enumerate(normal_samples[size]):
        theta1_est, theta2_est = method_of_moments_normal(sample)
        error1 = abs(theta1_est - theta_normal)
        error2 = abs(theta2_est - sigma_normal)
        print(f"Выборка {i + 1}: theta_1 = {theta1_est:.4f}, theta_2 = {theta2_est:.4f}, ошибки: err_theta_1 = {error1:.4f}, err_theta_2 = {error2:.4f}")

# Функция для ММП (Пуассон)
def mle_poisson(sample):
    return sum(sample) / len(sample)  # Совпадает с методом моментов

# Вычисление оценок ММП для распределения Пуассона
print("\nМЕТОД МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ - РАСПРЕДЕЛЕНИЕ ПУАССОНА")
print("Истинный параметр theta =", theta_poisson)

for size in sample_sizes:
    print(f"\nРазмер выборки: {size}")
    print("-" * 40)
    for i, sample in enumerate(poisson_samples[size]):
        estimate = mle_poisson(sample)
        error = abs(estimate - theta_poisson)
        print(f"Выборка {i+1}: theta = {estimate:.4f}, ошибка = {error:.4f}")

# Функция для ММП (нормальное распределение)
def mle_normal(sample):
    n = len(sample)
    theta1 = sum(sample) / n
    theta2 = math.sqrt(sum((x - theta1)**2 for x in sample) / n)
    return theta1, theta2

# Вычисление оценок ММП для нормального распределения
print("\nМЕТОД МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ - НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ")
print("Истинные параметры: theta_1 =", theta_normal, ", theta_2 =", sigma_normal)

for size in sample_sizes:
    print(f"\nРазмер выборки: {size}")
    print("-" * 60)
    for i, sample in enumerate(normal_samples[size]):
        theta1_est, theta2_est = mle_normal(sample)
        error1 = abs(theta1_est - theta_normal)
        error2 = abs(theta2_est - sigma_normal)
        print(f"Выборка {i+1}: theta_1 = {theta1_est:.4f}, theta_2 = {theta2_est:.4f}, ошибки: err_theta_1 = {error1:.4f}, err_theta_2 = {error2:.4f}")

#theta
def optimal_estimate_poisson_theta(sample):
    return sum(sample) / len(sample)

#theta^2
def optimal_estimate_poisson_theta_squared(sample):
    n = len(sample)
    x_bar = sum(sample) / n
    return x_bar**2 - x_bar / n

#e^{-theta}
def optimal_estimate_poisson_exp_theta(sample):
    n = len(sample)
    total = sum(sample)
    log_estimate = total * math.log(1 - 1 / n)
    return math.exp(log_estimate)

#Информация Фишера для распределения Пуассона
def calculate_fisher_information_poisson(theta, n):
    return n / theta

#Нижняя граница Рао-крамера
def rao_cramer_bound_poisson(theta, n, tau_derivative):
    fisher_info = calculate_fisher_information_poisson(theta, n)
    return (tau_derivative ** 2) / fisher_info

#Для одной выборки
def analyze_poisson_sample(sample, true_theta):
    n = len(sample)

    # Вычисление оценок
    theta_est = optimal_estimate_poisson_theta(sample)
    theta_sq_est = optimal_estimate_poisson_theta_squared(sample)
    exp_theta_est = optimal_estimate_poisson_exp_theta(sample)

    # Теоретические значения
    true_theta_sq = true_theta ** 2
    true_exp_theta = math.exp(-true_theta)

    # Ошибки оценок
    error_theta = abs(theta_est - true_theta)
    error_theta_sq = abs(theta_sq_est - true_theta_sq)
    error_exp_theta = abs(exp_theta_est - true_exp_theta)

    # Границы Рао-Крамера
    rc_bound_theta = rao_cramer_bound_poisson(true_theta, n, 1)  # производная θ равна 1
    rc_bound_theta_sq = rao_cramer_bound_poisson(true_theta, n, 2 * true_theta)  # производная θ²
    rc_bound_exp_theta = rao_cramer_bound_poisson(true_theta, n, -math.exp(-true_theta))  # производная e^{-θ}

    return {
        'theta_est': theta_est, 'error_theta': error_theta, 'rc_bound_theta': rc_bound_theta,
        'theta_sq_est': theta_sq_est, 'error_theta_sq': error_theta_sq, 'rc_bound_theta_sq': rc_bound_theta_sq,
        'exp_theta_est': exp_theta_est, 'error_exp_theta': error_exp_theta, 'rc_bound_exp_theta': rc_bound_exp_theta
    }

#Для всех выборок
def calculate_all_poisson_estimates(poisson_samples, true_theta):
    results = {}

    for size, samples_list in poisson_samples.items():
        results[size] = []
        for i, sample in enumerate(samples_list):
            analysis = analyze_poisson_sample(sample, true_theta)

            results[size].append({
                'sample_num': i + 1,
                'sample': sample,
                'theta_estimate': analysis['theta_est'],
                'theta_squared_estimate': analysis['theta_sq_est'],
                'exp_theta_estimate': analysis['exp_theta_est'],
                'errors': {
                    'theta': analysis['error_theta'],
                    'theta_squared': analysis['error_theta_sq'],
                    'exp_theta': analysis['error_exp_theta']
                },
                'rc_bounds': {
                    'theta': analysis['rc_bound_theta'],
                    'theta_squared': analysis['rc_bound_theta_sq'],
                    'exp_theta': analysis['rc_bound_exp_theta']
                }
            })

    return results

# Вычисление оценок для всех выборок
poisson_optimal_estimates = calculate_all_poisson_estimates(poisson_samples, theta_poisson)

# Вывод результатов
print("\nОПТИМАЛЬНЫЕ ОЦЕНКИ ДЛЯ РАСПРЕДЕЛЕНИЯ ПУАССОНА")
print(f"Истинный параметр theta = {theta_poisson}")

for size in poisson_optimal_estimates:
    print(f"\nРазмер выборки: {size}")
    print(f"Информация Фишера: i_n(θ) = {calculate_fisher_information_poisson(theta_poisson, size):.4f}")
    print("-" * 50)

    for result in poisson_optimal_estimates[size]:
        print(f"Выборка {result['sample_num']}:")
        print(f"  θ: {result['theta_estimate']:.4f} (ошибка: {result['errors']['theta']:.4f})")
        print(f"    Нижняя граница Рао-Крамера: {result['rc_bounds']['theta']:.6f}")

        print(f"  θ²: {result['theta_squared_estimate']:.4f} (ошибка: {result['errors']['theta_squared']:.4f})")
        print(f"    Нижняя граница Рао-Крамера: {result['rc_bounds']['theta_squared']:.6f}")

        print(f"  e^(-θ): {result['exp_theta_estimate']:.10e} (ошибка: {result['errors']['exp_theta']:.10e})")
        print(f"    Нижняя граница Рао-Крамера: {result['rc_bounds']['exp_theta']:.10e}")

def optimal_estimate_normal_theta(sample):
    return sum(sample) / len(sample)

def optimal_estimate_normal_theta_squared(sample):
    n = len(sample)
    y_bar = sum(sample) / n
    s2 = statistics.variance(sample)
    return y_bar**2 - s2 / n

def optimal_estimate_normal_sigma_squared(sample):
    return statistics.variance(sample)

def calculate_fisher_information_normal(sigma, n):
    return n / (2 * sigma**4)

def rao_cramer_bound_normal(sigma, n, tau_derivative):
    fisher_info = calculate_fisher_information_normal(sigma, n)
    return (tau_derivative ** 2) / fisher_info

# Пример использования для одной выборки
def analyze_normal_sample(sample, true_theta, true_sigma):
    n = len(sample)

    # Вычисление оценок
    theta_est = optimal_estimate_normal_theta(sample)
    theta_sq_est = optimal_estimate_normal_theta_squared(sample)
    sigma_sq_est = optimal_estimate_normal_sigma_squared(sample)

    # Теоретические значения
    true_theta_sq = true_theta ** 2
    true_sigma_sq = true_sigma ** 2

    # Ошибки оценок
    error_theta = abs(theta_est - true_theta)
    error_theta_sq = abs(theta_sq_est - true_theta_sq)
    error_sigma_sq = abs(sigma_sq_est - true_sigma_sq)

    # Границы Крамера-Рао
    rc_bound_theta = rao_cramer_bound_normal(true_sigma, n, 1)
    rc_bound_theta_sq = rao_cramer_bound_normal(true_sigma, n, 2 * true_theta)
    rc_bound_sigma_sq = rao_cramer_bound_normal(true_sigma, n, 1)

    return {
        'theta_est': theta_est, 'error_theta': error_theta,
        'rc_bound_theta': rc_bound_theta, 'theta_sq_est': theta_sq_est,
        'error_theta_sq': error_theta_sq, 'rc_bound_theta_sq': rc_bound_theta_sq,
        'sigma_sq_est': sigma_sq_est, 'error_sigma_sq': error_sigma_sq,
        'rc_bound_sigma_sq': rc_bound_sigma_sq
    }


def calculate_all_normal_estimates(normal_samples, true_theta, true_sigma):
    results = {}

    for size, samples_list in normal_samples.items():
        results[size] = []
        for i, sample in enumerate(samples_list):
            analysis = analyze_normal_sample(sample, true_theta, true_sigma)

            results[size].append({
                'sample_num': i + 1,
                'sample': sample,
                'theta_estimate': analysis['theta_est'],
                'theta_squared_estimate': analysis['theta_sq_est'],
                'sigma_squared_estimate': analysis['sigma_sq_est'],
                'errors': {
                    'theta': analysis['error_theta'],
                    'theta_squared': analysis['error_theta_sq'],
                    'sigma_squared': analysis['error_sigma_sq'],
                },
                'rc_bounds': {
                    'theta': analysis['rc_bound_theta'],
                    'theta_squared': analysis['rc_bound_theta_sq'],
                    'sigma_squared': analysis['rc_bound_sigma_sq']
                }
            })

    return results


# Вычисление оценок для всех выборок
normal_optimal_estimates = calculate_all_normal_estimates(normal_samples, theta_normal, sigma_normal)

# Вывод результатов
print("\nОПТИМАЛЬНЫЕ ОЦЕНКИ ДЛЯ НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ")
print(f"Истинные параметры: theta = {theta_normal}, sigma = {sigma_normal}")

for size in normal_optimal_estimates:
    print(f"\nРазмер выборки: {size}")
    print(f"Информация Фишера для sigma^2: i_n(sigma^2) = {calculate_fisher_information_normal(sigma_normal, size):.4f}")
    print("-" * 60)

    for result in normal_optimal_estimates[size]:
        print(f"Выборка {result['sample_num']}:")

        print(f"theta: {result['theta_estimate']:.6f} (ошибка: {result['errors']['theta']:.6f})")
        print(f"Нижняя граница Рао-Крамера: {result['rc_bounds']['theta']:.6e}")
        print()

        print(f"theta^2: {result['theta_squared_estimate']:.6f} (ошибка: {result['errors']['theta_squared']:.6f})")
        print(f"Нижняя граница Рао-Крамера: {result['rc_bounds']['theta_squared']:.6e}")
        print()

        print(f"sigma^2: {result['sigma_squared_estimate']:.6f} (ошибка: {result['errors']['sigma_squared']:.6f})")
        print(f"Нижняя граница Рао-Крамера: {result['rc_bounds']['sigma_squared']:.6e}")
        print()

