import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, shapiro
import scipy.stats as stats

# РАСПРЕДЕЛЕНИЕ ПУАССОНА

print("РАСПРЕДЕЛЕНИЕ ПУАССОНА")

# Параметры
theta0_poisson = 17.5
theta1_poisson = 19.0
alpha = 0.05
beta = 0.10

# Квантили нормального распределения
z_alpha = norm.ppf(1 - alpha)  # z_{1-alpha}
z_beta = norm.ppf(1 - beta)    # z_{1-beta}

print(f"alpha = {alpha}, beta = {beta}")
print(f"z_{1-alpha} = {z_alpha:.4f}, z_{1-beta} = {z_beta:.4f}")

# Вычисление параметров
log_ratio = np.log(theta1_poisson / theta0_poisson)
print(f"\nln(theta1/theta2) = ln({theta1_poisson}/{theta0_poisson}) = {log_ratio:.6f}")

mu0 = theta0_poisson * log_ratio - (theta1_poisson - theta0_poisson)
sigma0_sq = theta0_poisson * (log_ratio ** 2)
sigma0 = np.sqrt(sigma0_sq)

mu1 = theta1_poisson * log_ratio - (theta1_poisson - theta0_poisson)
sigma1_sq = theta1_poisson * (log_ratio ** 2)
sigma1 = np.sqrt(sigma1_sq)

print(f"\nПараметры при H0 (theta={theta0_poisson}):")
print(f"  mu0 = {mu0:.6f}")
print(f"  sigma0^2 = {sigma0_sq:.6f}")
print(f"  sigma0 = {sigma0:.6f}")

print(f"\nПараметры при H1 (theta={theta1_poisson}):")
print(f"  mu1 = {mu1:.6f}")
print(f"  sigma1^2 = {sigma1_sq:.6f}")
print(f"  sigma1 = {sigma1:.6f}")

# Вычисление необходимого объема выборки
n_poisson = ((z_alpha * sigma0 + z_beta * sigma1) / (mu1 - mu0)) ** 2
n_poisson = int(np.ceil(n_poisson))

print(f"\nМинимальный необходимый объем выборки:")
print(f"  sqrt(n) = (z_alpha * sigma0 + z_beta * sigma_1) / (mu1 - mu0)")
print(f"     = ({z_alpha:.3f} * {sigma0:.4f} + {z_beta:.3f} * {sigma1:.4f}) / ({mu1:.4f} - {mu0:.4f})")
print(f"     = {z_alpha * sigma0 + z_beta * sigma1:.4f} / {mu1 - mu0:.4f}")
print(f"     = {np.sqrt(n_poisson):.3f}")
print(f"  n = {n_poisson}")

# Порог критической области
ln_c_poisson = n_poisson * mu0 + z_alpha * np.sqrt(n_poisson) * sigma0
print(f"\nПорог критической области:")
print(f"  ln(c) = n * mu0 + z_alpha * sqrt(n) * sigma0")
print(f"        = {n_poisson} * {mu0:.6f} + {z_alpha:.3f} * sqrt({n_poisson}) * {sigma0:.6f}")
print(f"        = {ln_c_poisson:.4f}")

# Функция для вычисления логарифма отношения правдоподобия
def ln_l_poisson(sample, theta0, theta1):
    S = np.sum(sample)
    log_ratio = np.log(theta1 / theta0)
    return S * log_ratio - len(sample) * (theta1 - theta0)

# Генерация выборок и проверка
np.random.seed(42)
sample_H0_poisson = poisson.rvs(theta0_poisson, size=n_poisson)
sample_H1_poisson = poisson.rvs(theta1_poisson, size=n_poisson)

ln_l_H0 = ln_l_poisson(sample_H0_poisson, theta0_poisson, theta1_poisson)
ln_l_H1 = ln_l_poisson(sample_H1_poisson, theta0_poisson, theta1_poisson)

print(f"\nПроверка на реальных выборках:")
print(f"  Для выборки из H0: ln l = {ln_l_H0:.4f}")
print(f"    Решение: {'ОТВЕРГАЕМ H0' if ln_l_H0 > ln_c_poisson else 'ПРИНИМАЕМ H0'}")
print(f"  Для выборки из H1: ln l = {ln_l_H1:.4f}")
print(f"    Решение: {'ОТВЕРГАЕМ H0' if ln_l_H1 > ln_c_poisson else 'ПРИНИМАЕМ H0'}")

# Проверка нормальности статистики с помощью моделирования
n_simulations = 5000
stats_H0_poisson = []
stats_H1_poisson = []

for _ in range(n_simulations):
    sample_H0 = poisson.rvs(theta0_poisson, size=n_poisson)
    sample_H1 = poisson.rvs(theta1_poisson, size=n_poisson)
    stats_H0_poisson.append(ln_l_poisson(sample_H0, theta0_poisson, theta1_poisson))
    stats_H1_poisson.append(ln_l_poisson(sample_H1, theta0_poisson, theta1_poisson))

# Тест Шапиро-Уилка на нормальность
shapiro_stat_H0, shapiro_p_H0 = shapiro(stats_H0_poisson)
shapiro_stat_H1, shapiro_p_H1 = shapiro(stats_H1_poisson)

print(f"\nПроверка нормальности статистики (Шапиро-Уилк):")
print(f"  Для H0: статистика = {shapiro_stat_H0:.4f}, p-value = {shapiro_p_H0:.4f}")
print(f"  Для H1: статистика = {shapiro_stat_H1:.4f}, p-value = {shapiro_p_H1:.4f}")

# Построение графиков для распределения Пуассона
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Распределение Пуассона: проверка гипотез theta0={theta0_poisson}, theta1={theta1_poisson}, n={n_poisson}', fontsize=14)

# Гистограмма для H0
axes[0].hist(stats_H0_poisson, bins=30, density=True, alpha=0.6, color='blue', label='Эмпирическое (H0)')
x_min0, x_max0 = min(stats_H0_poisson), max(stats_H0_poisson)
x0 = np.linspace(x_min0, x_max0, 1000)
pdf_H0 = norm.pdf(x0, loc=n_poisson*mu0, scale=np.sqrt(n_poisson)*sigma0)
axes[0].plot(x0, pdf_H0, 'b-', linewidth=2, label=f'Теоретическое N({n_poisson*mu0:.1f}, {n_poisson*sigma0_sq:.1f})')
axes[0].axvline(ln_c_poisson, color='red', linestyle='--', linewidth=2, label=f'Порог ln(c)={ln_c_poisson:.2f}')
axes[0].set_xlabel('ln l(X)')
axes[0].set_ylabel('Плотность вероятности')
axes[0].set_title(f'Распределение ln l при H0 (theta={theta0_poisson})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Гистограмма для H1
axes[1].hist(stats_H1_poisson, bins=30, density=True, alpha=0.6, color='orange', label='Эмпирическое (H1)')
x_min1, x_max1 = min(stats_H1_poisson), max(stats_H1_poisson)
x1 = np.linspace(x_min1, x_max1, 1000)
pdf_H1 = norm.pdf(x1, loc=n_poisson*mu1, scale=np.sqrt(n_poisson)*sigma1)
axes[1].plot(x1, pdf_H1, 'orange', linewidth=2, label=f'Теоретическое N({n_poisson*mu1:.1f}, {n_poisson*sigma1_sq:.1f})')
axes[1].axvline(ln_c_poisson, color='red', linestyle='--', linewidth=2, label=f'Порог ln(c)={ln_c_poisson:.2f}')
axes[1].set_xlabel('ln l(X)')
axes[1].set_ylabel('Плотность вероятности')
axes[1].set_title(f'Распределение ln l при H0 (theta={theta1_poisson})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ

print("\nНОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ")

# Параметры
mu0_normal = 15.5
mu1_normal = 17.0
sigma_normal = 6.0

# Вычисление необходимого объема выборки
Delta = mu1_normal - mu0_normal
n_normal = ((z_alpha + z_beta) * sigma_normal / Delta) ** 2
n_normal = int(np.ceil(n_normal))

print(f"\nПараметры нормального распределения:")
print(f"  mu0 = {mu0_normal}, mu1 = {mu1_normal}, sigma = {sigma_normal}")
print(f"  Delta = mu1 - mu0 = {Delta}")

print(f"\nМинимальный необходимый объем выборки:")
print(f"  n = [(z_alpha + z_beta) * sigma / Delta]^2")
print(f"    = [({z_alpha:.3f} + {z_beta:.3f}) * {sigma_normal} / {Delta}]^2")
print(f"    = [{z_alpha + z_beta:.3f}·{sigma_normal} / {Delta}]²")
print(f"    = [{(z_alpha + z_beta) * sigma_normal:.3f} / {Delta}]²")
print(f"    = [{((z_alpha + z_beta) * sigma_normal / Delta):.3f}]²")
print(f"    = {n_normal}")

# Параметры распределения ln l для нормального распределения
mu0_star = -Delta**2 / (2 * sigma_normal**2)
mu1_star = Delta**2 / (2 * sigma_normal**2)
sigma_star = Delta / sigma_normal

print(f"\nПараметры распределения ln l:")
print(f"  mu0* = -Delta^2/(2sigma^2) = -{Delta**2}/(2 * {sigma_normal**2}) = {mu0_star:.6f}")
print(f"  mu1* = Delta^2/(2sigma^2) = {Delta**2}/(2 * {sigma_normal**2}) = {mu1_star:.6f}")
print(f"  sigma* = Delta/sigma = {Delta}/{sigma_normal} = {sigma_star:.6f}")

# Порог критической области
ln_c_normal = n_normal * mu0_star + z_alpha * np.sqrt(n_normal) * sigma_star
print(f"\nПорог критической области:")
print(f"  ln(c) = n * mu0* + z_alpha * sqrt(n) * sigma*")
print(f"        = {n_normal}·{mu0_star:.6f} + {z_alpha:.3f} * sqrt({n_normal}) * {sigma_star:.6f}")
print(f"        = {ln_c_normal:.4f}")

# Функция для вычисления логарифма отношения правдоподобия
def ln_l_normal(sample, mu0, mu1, sigma):
    S = np.sum(sample)
    Delta = mu1 - mu0
    return (Delta / sigma**2) * S - (len(sample) / (2 * sigma**2)) * (mu1**2 - mu0**2)

# Генерация выборок и проверка
np.random.seed(123)
sample_H0_normal = np.random.normal(mu0_normal, sigma_normal, n_normal)
sample_H1_normal = np.random.normal(mu1_normal, sigma_normal, n_normal)

ln_l_H0_norm = ln_l_normal(sample_H0_normal, mu0_normal, mu1_normal, sigma_normal)
ln_l_H1_norm = ln_l_normal(sample_H1_normal, mu0_normal, mu1_normal, sigma_normal)

print(f"\nПроверка на реальных выборках:")
print(f"  Для выборки из H0: ln l = {ln_l_H0_norm:.4f}")
print(f"    Решение: {'ОТВЕРГАЕМ H0' if ln_l_H0_norm > ln_c_normal else 'ПРИНИМАЕМ H0'}")
print(f"  Для выборки из H1: ln l = {ln_l_H1_norm:.4f}")
print(f"    Решение: {'ОТВЕРГАЕМ H0' if ln_l_H1_norm > ln_c_normal else 'ПРИНИМАЕМ H0'}")

# Проверка нормальности статистики с помощью моделирования
stats_H0_normal = []
stats_H1_normal = []

for _ in range(n_simulations):
    sample_H0 = np.random.normal(mu0_normal, sigma_normal, n_normal)
    sample_H1 = np.random.normal(mu1_normal, sigma_normal, n_normal)
    stats_H0_normal.append(ln_l_normal(sample_H0, mu0_normal, mu1_normal, sigma_normal))
    stats_H1_normal.append(ln_l_normal(sample_H1, mu0_normal, mu1_normal, sigma_normal))

# Тест Шапиро-Уилка на нормальность
shapiro_stat_H0_norm, shapiro_p_H0_norm = shapiro(stats_H0_normal)
shapiro_stat_H1_norm, shapiro_p_H1_norm = shapiro(stats_H1_normal)

print(f"\nПроверка нормальности статистики (Шапиро-Уилк):")
print(f"  Для H0: статистика = {shapiro_stat_H0_norm:.4f}, p-value = {shapiro_p_H0_norm:.4f}")
print(f"  Для H1: статистика = {shapiro_stat_H1_norm:.4f}, p-value = {shapiro_p_H1_norm:.4f}")

# Построение графиков для нормального распределения
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Нормальное распределение: проверка гипотез mu0={mu0_normal}, mu1={mu1_normal}, sigma={sigma_normal}, n={n_normal}', fontsize=14)

# Гистограмма для H0
axes[0].hist(stats_H0_normal, bins=30, density=True, alpha=0.6, color='blue', label='Эмпирическое (H0)')
x_min0_norm, x_max0_norm = min(stats_H0_normal), max(stats_H0_normal)
x0_norm = np.linspace(x_min0_norm, x_max0_norm, 1000)
pdf_H0_norm = norm.pdf(x0_norm, loc=n_normal*mu0_star, scale=np.sqrt(n_normal)*sigma_star)
axes[0].plot(x0_norm, pdf_H0_norm, 'b-', linewidth=2, label=f'Теоретическое N({n_normal*mu0_star:.1f}, {n_normal*sigma_star**2:.1f})')
axes[0].axvline(ln_c_normal, color='red', linestyle='--', linewidth=2, label=f'Порог ln(c)={ln_c_normal:.2f}')
axes[0].set_xlabel('ln l(X)')
axes[0].set_ylabel('Плотность вероятности')
axes[0].set_title(f'Распределение ln l при H0 (mu={mu0_normal})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Гистограмма для H1
axes[1].hist(stats_H1_normal, bins=30, density=True, alpha=0.6, color='orange', label='Эмпирическое (H1)')
x_min1_norm, x_max1_norm = min(stats_H1_normal), max(stats_H1_normal)
x1_norm = np.linspace(x_min1_norm, x_max1_norm, 1000)
pdf_H1_norm = norm.pdf(x1_norm, loc=n_normal*mu1_star, scale=np.sqrt(n_normal)*sigma_star)
axes[1].plot(x1_norm, pdf_H1_norm, 'orange', linewidth=2, label=f'Теоретическое N({n_normal*mu1_star:.1f}, {n_normal*sigma_star**2:.1f})')
axes[1].axvline(ln_c_normal, color='red', linestyle='--', linewidth=2, label=f'Порог ln(c)={ln_c_normal:.2f}')
axes[1].set_xlabel('ln l(X)')
axes[1].set_ylabel('Плотность вероятности')
axes[1].set_title(f'Распределение ln l при H1 (mu={mu1_normal})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
