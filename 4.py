import numpy as np
import math
from scipy import stats
from scipy.stats import poisson, norm, chi2, kstwobign
from scipy.special import erf
import matplotlib.pyplot as plt
from collections import defaultdict
import math

# Параметры распределений
theta_poisson = 17.5
theta_normal = 15.5
sigma_normal = 6.0


# Генерация выборок
def poisson_sample(theta, size):
    sample = []
    for _ in range(size):
        k = 0
        p = 1.0
        L = math.exp(-theta)
        while p > L:
            k += 1
            U = np.random.uniform(0, 1)
            p *= U
        sample.append(k - 1)
    return sample


def normal_sample(theta, sigma, size):
    sample = []
    for _ in range(size):
        U1, U2 = np.random.uniform(0, 1, 2)
        Z = math.sqrt(-2 * math.log(U1)) * math.cos(2 * math.pi * U2)
        sample.append(theta + sigma * Z)
    return sample


# Генерация выборок
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
num_samples = 5

poisson_samples = {}
normal_samples = {}

for size in sample_sizes:
    poisson_samples[size] = [poisson_sample(theta_poisson, size) for _ in range(num_samples)]
    normal_samples[size] = [normal_sample(theta_normal, sigma_normal, size) for _ in range(num_samples)]


# Вычисление статистики Колмогорова-Смирнова
def compute_ks_statistic(sample, cdf_func, params=None):
    n = len(sample)
    sample_sorted = np.sort(sample)

    # Эмпирическая функция распределения
    i_over_n = np.arange(1, n + 1) / n
    i_minus_1_over_n = np.arange(0, n) / n

    # Теоретическая функция распределения
    if params is None:
        F_theoretical = cdf_func(sample_sorted)
    else:
        F_theoretical = cdf_func(sample_sorted, *params)

    # Вычисление D_n^+ и D_n^-
    D_plus = np.max(np.abs(i_over_n - F_theoretical))
    D_minus = np.max(np.abs(F_theoretical - i_minus_1_over_n))
    D_n = max(D_plus, D_minus)

    # Статистика с поправкой Большева
    S = (6 * n * D_n + 1) / (6 * math.sqrt(n))

    return D_n, S


# Критерий Колмогорова-Смирнова для простой гипотезы
def ks_test_simple(sample, distribution, alpha=0.05):
    if distribution == 'poisson':
        # Теоретическая ФР Пуассона
        def theoretical_cdf(x):
            return poisson.cdf(x, theta_poisson)
    elif distribution == 'normal':
        # Теоретическая ФР нормального распределения
        def theoretical_cdf(x):
            return 0.5 * (1 + erf((x - theta_normal) / (sigma_normal * math.sqrt(2))))

    D_n, S = compute_ks_statistic(sample, theoretical_cdf)

    # Критическое значение распределения Колмогорова
    K_crit = kstwobign.ppf(1 - alpha)

    # p-value
    p_value = 1 - kstwobign.cdf(S)

    return {
        'D_n': D_n,
        'S': S,
        'critical_value': K_crit,
        'p_value': p_value,
        'reject': S > K_crit
    }


# Правило Старджесса для определения числа интервалов
def sturges_rule(n):
    return int(1 + 3.322 * math.log10(n))


# Вычисление статистики хи-квадрат Пирсона
def compute_chi2_statistic(sample, distribution, params=None, num_bins=None):
    n = len(sample)

    if num_bins is None:
        num_bins = max(5, sturges_rule(n))

    if distribution == 'poisson':
        # Для дискретного распределения используем уникальные значения
        unique_vals, observed = np.unique(sample, return_counts=True)

        # Теоретические вероятности
        if params is None:
            # Простая гипотеза
            theoretical_probs = poisson.pmf(unique_vals, theta_poisson)
        else:
            # Сложная гипотеза
            theta_est = params
            theoretical_probs = poisson.pmf(unique_vals, theta_est)

        # Объединяем категории с ожидаемой частотой < 5
        expected = n * theoretical_probs
        mask = expected >= 5

        # Если после объединения осталась только одна категория, возвращаем None
        if sum(mask) <= 1:
            return None

        observed = observed[mask]
        expected = expected[mask]
        theoretical_probs = theoretical_probs[mask]
        num_bins_adj = len(observed)

        # Число степеней свободы
        if params is None:
            df = num_bins_adj - 1  # Простая гипотеза
        else:
            df = num_bins_adj - 1 - 1  # Сложная гипотеза (1 оцененный параметр)

    elif distribution == 'normal':
        # Для непрерывного распределения строим гистограмму
        hist, bin_edges = np.histogram(sample, bins=num_bins, density=False)
        observed = hist

        # Теоретические вероятности
        if params is None:
            # Простая гипотеза
            cdf_values = norm.cdf(bin_edges, theta_normal, sigma_normal)
        else:
            # Сложная гипотеза
            theta_est, sigma_est = params
            cdf_values = norm.cdf(bin_edges, theta_est, sigma_est)

        theoretical_probs = np.diff(cdf_values)

        # Объединяем интервалы с ожидаемой частотой < 5
        expected = n * theoretical_probs
        mask = expected >= 5

        if sum(mask) <= 1:
            return None

        observed = observed[mask]
        expected = expected[mask]
        theoretical_probs = theoretical_probs[mask]
        num_bins_adj = len(observed)

        # Число степеней свободы
        if params is None:
            df = num_bins_adj - 1  # Простая гипотеза
        else:
            df = num_bins_adj - 1 - 2  # Сложная гипотеза

    # Вычисляем статистику хи-квадрат
    chi2_stat = np.sum((observed - expected) ** 2 / expected)

    return chi2_stat, df, observed, expected


# Критерий хи-квадрат для простой гипотезы
def chi_square_test_simple(sample, distribution, num_bins=None, alpha=0.05):
    result = compute_chi2_statistic(sample, distribution, params=None, num_bins=num_bins)

    if result is None:
        return None

    chi2_stat, df, observed, expected = result

    # Критическое значение и p-value
    chi2_crit = chi2.ppf(1 - alpha, df)
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return {
        'chi2_stat': chi2_stat,
        'df': df,
        'critical_value': chi2_crit,
        'p_value': p_value,
        'reject': chi2_stat > chi2_crit,
        'observed': observed,
        'expected': expected
    }


# Оценка ММП для распределения Пуассона
def mle_poisson(sample):
    return np.mean(sample)


# Оценка ММП для нормального распределения
def mle_normal(sample):
    theta_est = np.mean(sample)
    sigma_est = np.std(sample, ddof=0)  # Используем смещенную оценку для ММП
    return theta_est, sigma_est


# Критерий Колмогорова-Смирнова для сложной гипотезы с использованием метода Монте-Карло
def ks_test_complex(sample, distribution, alpha=0.05, n_simulations=5000):
    n = len(sample)

    if distribution == 'poisson':
        # Оценка параметра
        theta_est = mle_poisson(sample)

        # Вычисление статистики для исходной выборки
        def theoretical_cdf(x):
            return poisson.cdf(x, theta_est)

        D_n_observed, S_observed = compute_ks_statistic(sample, theoretical_cdf)

        # Монте-Карло симуляция для определения распределения статистики
        D_n_simulated = []
        for _ in range(n_simulations):
            # Генерация выборки с оцененным параметром
            sim_sample = np.random.poisson(theta_est, n)
            # Оценка параметра по симулированной выборке
            theta_sim = mle_poisson(sim_sample)

            # Вычисление статистики
            def sim_cdf(x):
                return poisson.cdf(x, theta_sim)

            D_n_sim, _ = compute_ks_statistic(sim_sample, sim_cdf)
            D_n_simulated.append(D_n_sim)

    elif distribution == 'normal':
        # Оценка параметров
        theta_est, sigma_est = mle_normal(sample)

        # Вычисление статистики для исходной выборки
        def theoretical_cdf(x):
            return 0.5 * (1 + erf((x - theta_est) / (sigma_est * math.sqrt(2))))

        D_n_observed, S_observed = compute_ks_statistic(sample, theoretical_cdf)

        # Монте-Карло симуляция
        D_n_simulated = []
        for _ in range(n_simulations):
            # Генерация выборки с оцененными параметрами
            sim_sample = np.random.normal(theta_est, sigma_est, n)
            # Оценка параметров по симулированной выборке
            theta_sim, sigma_sim = mle_normal(sim_sample)

            # Вычисление статистики
            def sim_cdf(x):
                return 0.5 * (1 + erf((x - theta_sim) / (sigma_sim * math.sqrt(2))))

            D_n_sim, _ = compute_ks_statistic(sim_sample, sim_cdf)
            D_n_simulated.append(D_n_sim)

    # Определение критического значения
    D_n_simulated = np.sort(D_n_simulated)
    critical_index = int((1 - alpha) * n_simulations)
    D_crit = D_n_simulated[critical_index]

    # p-value
    p_value = np.mean(np.array(D_n_simulated) > D_n_observed)

    return {
        'D_n': D_n_observed,
        'S': S_observed,
        'critical_value': D_crit,
        'p_value': p_value,
        'reject': D_n_observed > D_crit,
        'estimated_params': theta_est if distribution == 'poisson' else (theta_est, sigma_est)
    }


# Критерий хи-квадрат для сложной гипотезы
def chi_square_test_complex(sample, distribution, num_bins=None, alpha=0.05):
    # Оценка параметров методом максимального правдоподобия
    if distribution == 'poisson':
        params = mle_poisson(sample)
    elif distribution == 'normal':
        params = mle_normal(sample)

    # Вычисление статистики хи-квадрат с оцененными параметрами
    result = compute_chi2_statistic(sample, distribution, params=params, num_bins=num_bins)

    if result is None:
        return None

    chi2_stat, df, observed, expected = result

    # Критическое значение и p-value
    chi2_crit = chi2.ppf(1 - alpha, df)
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return {
        'chi2_stat': chi2_stat,
        'df': df,
        'critical_value': chi2_crit,
        'p_value': p_value,
        'reject': chi2_stat > chi2_crit,
        'observed': observed,
        'expected': expected,
        'estimated_params': params
    }


# Выполнение всех тестов для всех выборок
def perform_all_tests():
    results = {}

    for distribution in ['poisson', 'normal']:
        results[distribution] = {}

        for size in sample_sizes:
            results[distribution][size] = []

            samples = poisson_samples[size] if distribution == 'poisson' else normal_samples[size]

            for i, sample in enumerate(samples):
                # print(f"Обработка: {distribution}, n={size}, выборка {i + 1}/{len(samples)}")

                result = {
                    'sample_num': i + 1,
                    'size': size,
                    'simple_ks': ks_test_simple(sample, distribution),
                    'simple_chi2': chi_square_test_simple(sample, distribution),
                    'complex_ks': ks_test_complex(sample, distribution, n_simulations=1000),
                    'complex_chi2': chi_square_test_complex(sample, distribution)
                }
                results[distribution][size].append(result)

    return results


# Вывод результатов
def print_detailed_results(results):
    for distribution in ['poisson', 'normal']:
        print(f"\nРАСПРЕДЕЛЕНИЕ: {distribution.upper()}")

        for size in sample_sizes:
            print(f"\nРазмер выборки: n = {size}")

            for i, result in enumerate(results[distribution][size]):
                print(f"Выборка {i + 1}:")

                # Простая гипотеза - критерий Колмогорова-Смирнова
                ks_simple = result['simple_ks']
                if ks_simple:
                    print(f"  1. Критерий Колмогорова-Смирнова (простая гипотеза):")
                    print(f"     D_n = {ks_simple['D_n']:.6f}")
                    print(f"     S (с поправкой Большева) = {ks_simple['S']:.6f}")
                    print(f"     Критическое значение (alpha=0.05) = {ks_simple['critical_value']:.6f}")
                    print(f"     p-value = {ks_simple['p_value']:.6f}")
                    status = "ОТВЕРГАЕТСЯ" if ks_simple['reject'] else "ПРИНИМАЕТСЯ"
                    print(f"     Гипотеза: {status}")

                # Простая гипотеза - критерий хи-квадрат
                chi2_simple = result['simple_chi2']
                if chi2_simple:
                    print(f"\n  2. Критерий хи-квадрат Пирсона (простая гипотеза):")
                    print(f"     chi^2 = {chi2_simple['chi2_stat']:.6f}")
                    print(f"     Число степеней свободы = {chi2_simple['df']}")
                    print(f"     Критическое значение (alpha=0.05) = {chi2_simple['critical_value']:.6f}")
                    print(f"     p-value = {chi2_simple['p_value']:.6f}")
                    status = "ОТВЕРГАЕТСЯ" if chi2_simple['reject'] else "ПРИНИМАЕТСЯ"
                    print(f"     Гипотеза: {status}")

                # Сложная гипотеза - критерий Колмогорова-Смирнова
                ks_complex = result['complex_ks']
                if ks_complex:
                    print(f"\n  3. Критерий Колмогорова-Смирнова (сложная гипотеза):")
                    print(f"     D_n = {ks_complex['D_n']:.6f}")
                    if distribution == 'poisson':
                        print(f"     Оцененный параметр theta = {ks_complex['estimated_params']:.6f}")
                    else:
                        print(
                            f"     Оцененные параметры: theta = {ks_complex['estimated_params'][0]:.6f}, sigma = {ks_complex['estimated_params'][1]:.6f}")
                    print(f"     Критическое значение (alpha=0.05) = {ks_complex['critical_value']:.6f}")
                    print(f"     p-value = {ks_complex['p_value']:.6f}")
                    status = "ОТВЕРГАЕТСЯ" if ks_complex['reject'] else "ПРИНИМАЕТСЯ"
                    print(f"     Гипотеза: {status}")

                # Сложная гипотеза - критерий хи-квадрат
                chi2_complex = result['complex_chi2']
                if chi2_complex:
                    print(f"\n  4. Критерий хи-квадрат Пирсона (сложная гипотеза):")
                    print(f"     chi^2 = {chi2_complex['chi2_stat']:.6f}")
                    print(f"     Число степеней свободы = {chi2_complex['df']}")
                    if distribution == 'poisson':
                        print(f"     Оцененный параметр theta = {chi2_complex['estimated_params']:.6f}")
                    else:
                        print(
                            f"     Оцененные параметры: theta = {chi2_complex['estimated_params'][0]:.6f}, sigma = {chi2_complex['estimated_params'][1]:.6f}")
                    print(f"     Критическое значение (alpha=0.05) = {chi2_complex['critical_value']:.6f}")
                    print(f"     p-value = {chi2_complex['p_value']:.6f}")
                    status = "ОТВЕРГАЕТСЯ" if chi2_complex['reject'] else "ПРИНИМАЕТСЯ"
                    print(f"     Гипотеза: {status}")

                print()


# Создание сводной таблицы результатов
def create_summary_table(results):
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")

    for distribution in ['poisson', 'normal']:
        print(f"\n{distribution.upper()} РАСПРЕДЕЛЕНИЕ \n")

        # Заголовок таблицы
        header = f"{'Размер':<8} {'Выборка':<8} {'КС простая':<15} {'chi^2 простая':<15} {'КС сложная':<15} {'chi^2 сложная':<15}"
        print(header)

        for size in sample_sizes:
            for i, result in enumerate(results[distribution][size]):
                # Формируем строки для каждого теста
                ks_simple_reject = result['simple_ks']['reject'] if result['simple_ks'] else None
                chi2_simple_reject = result['simple_chi2']['reject'] if result['simple_chi2'] else None
                ks_complex_reject = result['complex_ks']['reject'] if result['complex_ks'] else None
                chi2_complex_reject = result['complex_chi2']['reject'] if result['complex_chi2'] else None

                # Символы для отображения результатов
                ks_simple_sym = "отвергаем" if ks_simple_reject else "принимаем" if ks_simple_reject is not None else "не применим"
                chi2_simple_sym = "отвергаем" if chi2_simple_reject else "принимаем" if chi2_simple_reject is not None else "не применим"
                ks_complex_sym = "отвергаем" if ks_complex_reject else "принимаем" if ks_complex_reject is not None else "не применим"
                chi2_complex_sym = "отвергаем" if chi2_complex_reject else "принимаем" if chi2_complex_reject is not None else "не применим"

                row = f"{size:<8} {i + 1:<8} {ks_simple_sym:<15} {chi2_simple_sym:<15} {ks_complex_sym:<15} {chi2_complex_sym:<15}"
                print(row)

        # Статистика по всем выборкам
        print("\nСТАТИСТИКА ОТВЕРЖЕНИЙ ПО ВСЕМ ВЫБОРКАМ:")

        total_counts = {
            'simple_ks': {'reject': 0, 'total': 0},
            'simple_chi2': {'reject': 0, 'total': 0},
            'complex_ks': {'reject': 0, 'total': 0},
            'complex_chi2': {'reject': 0, 'total': 0}
        }

        for size in sample_sizes:
            for result in results[distribution][size]:
                for test_type in total_counts.keys():
                    test_result = result[test_type]
                    if test_result:
                        total_counts[test_type]['total'] += 1
                        if test_result['reject']:
                            total_counts[test_type]['reject'] += 1

        for test_type, counts in total_counts.items():
            if counts['total'] > 0:
                reject_rate = counts['reject'] / counts['total'] * 100
                print(f"  {test_type}: {counts['reject']}/{counts['total']} ({reject_rate:.1f}%)")


# Построение графиков частот отвержения гипотез
def plot_rejection_rates(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Частота отвержения гипотез в зависимости от размера выборки', fontsize=16)

    test_configs = [
        ('simple_ks', 'Колмогорова-Смирнова\n(простая гипотеза)', 0, 0),
        ('simple_chi2', 'хи-квадрат\n(простая гипотеза)', 0, 1),
        ('complex_ks', 'Колмогорова-Смирнова\n(сложная гипотеза)', 1, 0),
        ('complex_chi2', 'хи-квадрат\n(сложная гипотеза)', 1, 1)
    ]

    for test_type, title, row, col in test_configs:
        ax = axes[row, col]

        for distribution, color, marker in [('poisson', 'blue', 'o'), ('normal', 'red', 's')]:
            rejection_rates = []
            sizes_clean = []

            for size in sample_sizes:
                reject_count = 0
                total_count = 0

                for result in results[distribution][size]:
                    test_result = result[test_type]
                    if test_result:
                        reject_count += 1 if test_result['reject'] else 0
                        total_count += 1

                if total_count > 0:
                    rejection_rate = reject_count / total_count
                    rejection_rates.append(rejection_rate)
                    sizes_clean.append(size)

            ax.plot(sizes_clean, rejection_rates, f'{marker}-',
                    color=color, linewidth=2, markersize=6,
                    label=f'{distribution.upper()}')

        ax.set_xlabel('Размер выборки (n)', fontsize=12)
        ax.set_ylabel('Частота отвержения', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
        ax.legend()
        ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.show()


# Сравнение мощности критериев
def plot_power_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Сравнение мощности критериев для разных распределений', fontsize=16)

    for idx, distribution in enumerate(['poisson', 'normal']):
        ax = axes[idx]

        # Собираем данные по всем тестам
        test_types = ['simple_ks', 'simple_chi2', 'complex_ks', 'complex_chi2']
        test_labels = ['KS простая', 'chi^2 простая', 'KS сложная', 'chi^2 сложная']

        # Вычисляем среднюю мощность по размерам выборок
        avg_power = {test: [] for test in test_types}
        sizes_list = []

        for size in sample_sizes:
            if size >= 100:  # Берем только достаточно большие выборки
                sizes_list.append(size)
                for test_type in test_types:
                    reject_count = 0
                    total_count = 0

                    for result in results[distribution][size]:
                        test_result = result[test_type]
                        if test_result:
                            reject_count += 1 if test_result['reject'] else 0
                            total_count += 1

                    if total_count > 0:
                        avg_power[test_type].append(reject_count / total_count)

        # Построение графиков
        for test_type, label in zip(test_types, test_labels):
            if len(avg_power[test_type]) == len(sizes_list):
                ax.plot(sizes_list, avg_power[test_type], 'o-', linewidth=2, markersize=6, label=label)

        ax.set_xlabel('Размер выборки (n)', fontsize=12)
        ax.set_ylabel('Мощность критерия', fontsize=12)
        ax.set_title(f'{distribution.upper()} распределение', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.legend()
        ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.show()


# Выполнение всех тестов
results = perform_all_tests()

# Вывод результатов
print("РЕЗУЛЬТАТЫ:")
print_detailed_results(results)

# Сводная таблица
create_summary_table(results)

# Графики
plot_rejection_rates(results)
plot_power_comparison(results)

# ОДНОРОДНОСТЬ ВЫБОРОК

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
        f"\n{'Пара (n,m)':<12} {'Среднее':<12} {'Станд. откл.':<12} {'Минимум':<12} {'Максимум':<12}")
    print(f"{'-' * 70}")

    for pair in sorted_pairs:
        stats = D_results[pair]
        n, m = pair

        mean_val = f"{stats['mean']:.4f}"
        std_val = f"{stats['std']:.4f}"
        min_val = f"{stats['min']:.4f}"
        max_val = f"{stats['max']:.4f}"

        print(f"({n},{m})   {mean_val:<12} {std_val:<12} {min_val:<12} {max_val:<12}")


print_D_statistics(D_poisson)
print_D_statistics(D_normal)


# Вычисление статистики Колмогорова-Смирнова
def compute_homogeneity_statistics(samples_dict, sample_sizes):
    homogeneity_stats = {}

    for n in sample_sizes:
        if n in samples_dict:
            for m in sample_sizes:
                if m in samples_dict and m >= n:  # Избегаем дублирования
                    key = (n, m)
                    homogeneity_stats[key] = {
                        'D_values': [],  # Значения D_{n,m}
                        'KS_stats': [],  # Статистики Колмогорова-Смирнова
                        'p_values': [],  # p-values
                        'rejections': []  # Флаги отвержения гипотезы
                    }

                    # Для каждой пары выборок размера n и m
                    for i, sample_n in enumerate(samples_dict[n]):
                        for j, sample_m in enumerate(samples_dict[m]):
                            if n == m and i >= j:
                                # Пропускаем дубликаты для одинаковых размеров
                                continue

                            # Вычисляем D_{n,m}
                            D_nm = compute_supremum_difference(sample_n, sample_m)

                            # Вычисляем статистику Колмогорова-Смирнова
                            KS_stat = math.sqrt((n * m) / (n + m)) * D_nm

                            # Вычисляем p-value (асимптотическое)
                            p_value = 1 - kstwobign.cdf(KS_stat)

                            # Проверка гипотезы на уровне α=0.05
                            critical_value = kstwobign.ppf(0.95)  # Для α=0.05
                            reject = KS_stat > critical_value

                            homogeneity_stats[key]['D_values'].append(D_nm)
                            homogeneity_stats[key]['KS_stats'].append(KS_stat)
                            homogeneity_stats[key]['p_values'].append(p_value)
                            homogeneity_stats[key]['rejections'].append(reject)

                    # Вычисляем средние значения для данной пары (n,m)
                    if homogeneity_stats[key]['D_values']:
                        homogeneity_stats[key]['mean_D'] = np.mean(homogeneity_stats[key]['D_values'])
                        homogeneity_stats[key]['mean_KS'] = np.mean(homogeneity_stats[key]['KS_stats'])
                        homogeneity_stats[key]['mean_p'] = np.mean(homogeneity_stats[key]['p_values'])
                        homogeneity_stats[key]['rejection_rate'] = np.mean(homogeneity_stats[key]['rejections'])

    return homogeneity_stats

# Вычисление статистики хи-квадрат
def compute_chi2_homogeneity_statistics(samples_dict, sample_sizes, num_bins='auto'):
    chi2_stats = {}

    for n in sample_sizes:
        if n in samples_dict and len(samples_dict[n]) >= 2:
            # Берем первые две выборки данного размера
            sample1 = samples_dict[n][0]
            sample2 = samples_dict[n][1]

            # Объединяем выборки для определения границ интервалов
            combined = np.concatenate([sample1, sample2])

            # Определяем число интервалов по правилу Старджесса
            if num_bins == 'auto':
                num_bins = max(5, int(1 + 3.322 * np.log10(len(combined))))

            # Создаем гистограммы
            hist1, bin_edges = np.histogram(sample1, bins=num_bins, density=False)
            hist2, _ = np.histogram(sample2, bins=bin_edges, density=False)

            # Проверяем, достаточно ли наблюдений в каждом интервале
            # Объединяем соседние интервалы, если ожидаемая частота < 5
            observed = np.vstack([hist1, hist2])
            row_totals = observed.sum(axis=1)
            col_totals = observed.sum(axis=0)
            total = observed.sum()

            # Вычисляем ожидаемые частоты
            expected = np.outer(row_totals, col_totals) / total

            # Объединяем интервалы, где ожидаемая частота < 5
            valid_cols = np.where(expected.min(axis=0) >= 5)[0]

            if len(valid_cols) <= 1:
                # Недостаточно интервалов для проверки
                chi2_stats[n] = None
                continue

            # Фильтруем только валидные интервалы
            observed_valid = observed[:, valid_cols]
            expected_valid = expected[:, valid_cols]

            # Вычисляем статистику хи-квадрат
            chi2_stat = np.sum((observed_valid - expected_valid) ** 2 / expected_valid)

            # Число степеней свободы
            # (k-1)(N-1), где k=2 (две выборки), N - число интервалов
            df = (2 - 1) * (len(valid_cols) - 1)

            # p-value
            p_value = 1 - chi2.cdf(chi2_stat, df)

            # Критическое значение для α=0.05
            critical_value = chi2.ppf(0.95, df)

            # Проверка гипотезы
            reject = chi2_stat > critical_value

            chi2_stats[n] = {
                'chi2_stat': chi2_stat,
                'df': df,
                'p_value': p_value,
                'critical_value': critical_value,
                'reject': reject,
                'num_bins_original': num_bins,
                'num_bins_valid': len(valid_cols)
            }

    return chi2_stats


# Визуализация результатов
def plot_homogeneity_analysis(homogeneity_poisson, homogeneity_normal,
                              chi2_poisson, chi2_normal, sample_sizes):
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Средние значения D_{n,m} для Пуассона
    ax1 = axes[0, 0]
    sizes_poisson = []
    mean_D_poisson = []

    for (n, m), stats in homogeneity_poisson.items():
        if n == m and 'mean_D' in stats:  # Только для сравнения выборок одинакового размера
            sizes_poisson.append(n)
            mean_D_poisson.append(stats['mean_D'])

    if sizes_poisson:
        ax1.plot(sizes_poisson, mean_D_poisson, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Размер выборки (n)', fontsize=12)
        ax1.set_ylabel('Среднее D_{n,n}', fontsize=12)
        ax1.set_title('Распределение Пуассона: среднее значение статистики D_{n,n}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')

    # Средние значения D_{n,m} для нормального распределения
    ax2 = axes[0, 1]
    sizes_normal = []
    mean_D_normal = []

    for (n, m), stats in homogeneity_normal.items():
        if n == m and 'mean_D' in stats:
            sizes_normal.append(n)
            mean_D_normal.append(stats['mean_D'])

    if sizes_normal:
        ax2.plot(sizes_normal, mean_D_normal, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Размер выборки (n)', fontsize=12)
        ax2.set_ylabel('Среднее D_{n,n}', fontsize=12)
        ax2.set_title('Нормальное распределение: среднее значение статистики D_{n,n}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')

    # Частота отвержения гипотезы (KS критерий) для Пуассона
    ax3 = axes[1, 0]
    rejection_rates_poisson = []

    for size in sample_sizes:
        key = (size, size)
        if key in homogeneity_poisson and 'rejection_rate' in homogeneity_poisson[key]:
            rejection_rates_poisson.append(homogeneity_poisson[key]['rejection_rate'])
        else:
            rejection_rates_poisson.append(0)

    ax3.bar(range(len(sample_sizes)), rejection_rates_poisson, color='blue', alpha=0.7)
    ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Ожидаемый уровень (alpha=0.05)')
    ax3.set_xlabel('Размер выборки', fontsize=12)
    ax3.set_ylabel('Доля отвержения гипотезы', fontsize=12)
    ax3.set_title('Распределение Пуассона: частота отвержения гипотезы (критерий Смирнова)', fontsize=14)
    ax3.set_xticks(range(len(sample_sizes)))
    ax3.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Частота отвержения гипотезы (KS критерий) для нормального распределения
    ax4 = axes[1, 1]
    rejection_rates_normal = []

    for size in sample_sizes:
        key = (size, size)
        if key in homogeneity_normal and 'rejection_rate' in homogeneity_normal[key]:
            rejection_rates_normal.append(homogeneity_normal[key]['rejection_rate'])
        else:
            rejection_rates_normal.append(0)

    ax4.bar(range(len(sample_sizes)), rejection_rates_normal, color='red', alpha=0.7)
    ax4.axhline(y=0.05, color='blue', linestyle='--', linewidth=2, label='Ожидаемый уровень (alpha=0.05)')
    ax4.set_xlabel('Размер выборки', fontsize=12)
    ax4.set_ylabel('Доля отвержения гипотезы', fontsize=12)
    ax4.set_title('Нормальное распределение: частота отвержения гипотезы (критерий Смирнова)', fontsize=14)
    ax4.set_xticks(range(len(sample_sizes)))
    ax4.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Сравнение p-values для обоих распределений
    ax5 = axes[2, 0]
    p_values_poisson = []
    p_values_normal = []

    for size in sample_sizes:
        key_poisson = (size, size)
        key_normal = (size, size)

        if key_poisson in homogeneity_poisson and 'mean_p' in homogeneity_poisson[key_poisson]:
            p_values_poisson.append(homogeneity_poisson[key_poisson]['mean_p'])
        else:
            p_values_poisson.append(0.5)  # Нейтральное значение при отсутствии данных

        if key_normal in homogeneity_normal and 'mean_p' in homogeneity_normal[key_normal]:
            p_values_normal.append(homogeneity_normal[key_normal]['mean_p'])
        else:
            p_values_normal.append(0.5)

    x_pos = np.arange(len(sample_sizes))
    width = 0.35

    ax5.bar(x_pos - width / 2, p_values_poisson, width, label='Пуассон', color='blue', alpha=0.7)
    ax5.bar(x_pos + width / 2, p_values_normal, width, label='Нормальное', color='red', alpha=0.7)
    ax5.axhline(y=0.05, color='green', linestyle='--', linewidth=2, label='Уровень значимости alpha=0.05')
    ax5.set_xlabel('Размер выборки', fontsize=12)
    ax5.set_ylabel('Среднее p-value', fontsize=12)
    ax5.set_title('Сравнение p-values для критерия Смирнова', fontsize=14)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Результаты критерия хи-квадрат
    ax6 = axes[2, 1]
    chi2_stats_poisson = []
    chi2_stats_normal = []

    for size in sample_sizes:
        if size in chi2_poisson and chi2_poisson[size] is not None:
            chi2_stats_poisson.append(chi2_poisson[size]['chi2_stat'])
        else:
            chi2_stats_poisson.append(0)

        if size in chi2_normal and chi2_normal[size] is not None:
            chi2_stats_normal.append(chi2_normal[size]['chi2_stat'])
        else:
            chi2_stats_normal.append(0)

    ax6.plot(sample_sizes, chi2_stats_poisson, 'bo-', label='Пуассон', linewidth=2, markersize=6)
    ax6.plot(sample_sizes, chi2_stats_normal, 'ro-', label='Нормальное', linewidth=2, markersize=6)

    # Добавляем критические значения
    critical_values = [chi2.ppf(0.95, df) for df in range(5, 15)]
    ax6.axhline(y=np.mean(critical_values), color='green', linestyle='--',
                linewidth=2, label='Среднее критическое значение')

    ax6.set_xlabel('Размер выборки', fontsize=12)
    ax6.set_ylabel('Статистика chi^2', fontsize=12)
    ax6.set_title('Статистики критерия chi^2 для проверки однородности', fontsize=14)
    ax6.set_xscale('log')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Вывод результатов критерий Колмогорова-Смирнова
def print_detailed_homogeneity_results(homogeneity_stats, distribution_name, sample_sizes):
    print(f"КРИТЕРИЙ КОЛМОГОРОВА-СМИРНОВА: {distribution_name}")

    # Создаем таблицу для выборок одинакового размера
    print(f"\nСравнение выборок одинакового размера (Критерий Колмогорова-Смирнова)")
    print(
        f"{'Размер':<10} {'Среднее D':<15} {'Средняя статистика':<20} {'Среднее p-value':<15} {'Доля отвержения':<15}")
    print(f"{'-' * 80}")

    for size in sample_sizes:
        key = (size, size)
        if key in homogeneity_stats and 'mean_D' in homogeneity_stats[key]:
            stats = homogeneity_stats[key]
            print(f"{size:<10} {stats['mean_D']:<15.4f} {stats['mean_KS']:<20.4f} "
                  f"{stats['mean_p']:<15.4f} {stats['rejection_rate'] * 100:<15.2f}%")

    # Анализ ошибок I рода
    print(f"\n\nАНАЛИЗ ОШИБОК I РОДА:")
    print(f"Ошибка I рода - отвержение верной гипотезы H0.")
    print(f"Заданный уровень значимости: alpha = 0.05")
    print(f"Теоретическая частота отвержения при верной H0: 5%")

    # Вычисляем общую частоту отвержения
    total_rejections = 0
    total_comparisons = 0

    for key, stats in homogeneity_stats.items():
        if 'rejections' in stats:
            total_rejections += sum(stats['rejections'])
            total_comparisons += len(stats['rejections'])

    if total_comparisons > 0:
        overall_rejection_rate = total_rejections / total_comparisons
        print(f"\nОбщая частота отвержения гипотезы H0: {overall_rejection_rate * 100:.2f}%")
        print(f"Отклонение от ожидаемого уровня alpha: {abs(overall_rejection_rate - 0.05) * 100:.2f}%")


# Вывод результатов критерий хи-квадрат
def print_chi2_homogeneity_results(chi2_stats, distribution_name, sample_sizes):
    print(f"КРИТЕРИЙ ХИ-КВАДРАТ: {distribution_name}")

    print(f"\nКритерий хи-квадрат для проверки однородности")
    print(f"{'Размер':<10} {'Статистика chi^2':<15} {'Степени свободы':<20} {'p-value':<15} {'Решение':<15}")
    print(f"{'-' * 80}")

    for size in sample_sizes:
        if size in chi2_stats and chi2_stats[size] is not None:
            stats = chi2_stats[size]
            decision = "ОТВЕРГАЕМ" if stats['reject'] else "ПРИНИМАЕМ"
            print(f"{size:<10} {stats['chi2_stat']:<15.4f} {stats['df']:<20} "
                  f"{stats['p_value']:<15.4f} {decision:<15}")


def perform_homogeneity_analysis_full(poisson_samples, normal_samples, sample_sizes):
    print("ПРОВЕРКА ГИПОТЕЗЫ ОБ ОДНОРОДНОСТИ ВЫБОРОК")

    # Вычисление статистик для распределения Пуассона
    homogeneity_poisson = compute_homogeneity_statistics(poisson_samples, sample_sizes)
    chi2_poisson = compute_chi2_homogeneity_statistics(poisson_samples, sample_sizes)

    # Вычисление статистик для нормального распределения
    homogeneity_normal = compute_homogeneity_statistics(normal_samples, sample_sizes)
    chi2_normal = compute_chi2_homogeneity_statistics(normal_samples, sample_sizes)

    # Вывод подробных результатов
    print_detailed_homogeneity_results(homogeneity_poisson, "РАСПРЕДЕЛЕНИЕ ПУАССОНА", sample_sizes)
    print_chi2_homogeneity_results(chi2_poisson, "РАСПРЕДЕЛЕНИЕ ПУАССОНА", sample_sizes)

    print_detailed_homogeneity_results(homogeneity_normal, "НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ", sample_sizes)
    print_chi2_homogeneity_results(chi2_normal, "НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ", sample_sizes)

    # Визуализация результатов
    plot_homogeneity_analysis(homogeneity_poisson, homogeneity_normal, chi2_poisson, chi2_normal, sample_sizes)


perform_homogeneity_analysis_full(poisson_samples, normal_samples, sample_sizes)
