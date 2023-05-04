import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random

def generate_cpu_ssd_ram_freq_pair():
    cpu = random.choice([8, 10, 12, 16, 20, 24, 32, 40])
    cpu_freq_range = list(np.arange(2.0, 4.1, 0.1))

    if cpu >= 32:
        ssd = random.choice([3072, 3596, 4096, 5120, 6144, 8192])
        ram = random.choice([1024, 2048])
        cpu_freq = random.choice(cpu_freq_range[15:])
    else:
        ssd = random.choice([1024, 1536, 2048, 3072, 3596, 4096, 5120, 6144, 8192])
        ram = random.choice([64, 128, 256, 512])
        cpu_freq = random.choice(cpu_freq_range[:15])

    return cpu, ssd, ram, cpu_freq

def generate_bw(ssd):
    if ssd <= 2048:
        bw = random.choice([bw for bw in range(100, 10001, 100) if 100 * (bw // 100) >= 10 * ssd])
    elif 2048 < ssd <= 4096:
        bw = random.choice([bw for bw in range(100, 10001, 100) if 100 * (bw // 100) >= 20 * ssd])
    else:
        bw = random.choice([bw for bw in range(100, 10001, 100) if 100 * (bw // 100) >= 30 * ssd])
    return bw
def generate_rent(cpu, ram, ssd, cpu_freq, bw):
    base_rent = 0
    rent = base_rent + cpu * 2 + ram * 1 + ssd * 0.1 + cpu_freq * 2 + bw * 0.01
    rent = rent + random.uniform(-rent * 0.1, rent * 0.1)  # Добавляем большее случайное отклонение
    return round(rent)
def calculate_r_squared(data):
    model = smf.ols(formula='rent ~ cpu + ram + ssd + cpu_freq + bw', data=data).fit()
    return model.rsquared
def generate_random_data(min_r_squared=0.8):
    r_squared = 0
    while r_squared < min_r_squared:
        cpu_ssd_ram_freq_pairs = [generate_cpu_ssd_ram_freq_pair() for _ in range(50)]
        bw_values = [generate_bw(pair[3]) for pair in cpu_ssd_ram_freq_pairs]
        rent_values = [generate_rent(pair[0], pair[2], pair[1], pair[3], bw_values[idx]) for idx, pair in enumerate(cpu_ssd_ram_freq_pairs)]

        data = pd.DataFrame({
            'rent': rent_values,
            'cpu': [pair[0] for pair in cpu_ssd_ram_freq_pairs],
            'ram': [pair[2] for pair in cpu_ssd_ram_freq_pairs],
            'ssd': [pair[1] for pair in cpu_ssd_ram_freq_pairs],
            'cpu_freq': [pair[3] for pair in cpu_ssd_ram_freq_pairs],
            'bw': bw_values,
        })

        r_squared = calculate_r_squared(data)

    return data

def check_significance(data, alpha=0.05):
    model = smf.ols(formula='rent ~ cpu + ram + ssd + cpu_freq + bw', data=data).fit()
    significant_params = [param for param, pvalue in model.pvalues.items() if pvalue < alpha and param != 'Intercept']
    return significant_params

alpha = 0.05
max_iterations = 20000
iterations = 0
min_significant_params = 5

while iterations < max_iterations:
    random_data = generate_random_data()
    significant_params = check_significance(random_data, alpha)

    if len(significant_params) >= min_significant_params:
        print(f"Significant parameters found: {', '.join(significant_params)}")
        print(f"Iteration: {iterations}")
        print(random_data)
        break

    iterations += 1

if iterations == max_iterations:
    print("Failed to find at least", min_significant_params, "significant parameters after", max_iterations, "iterations")
if len(significant_params) >= min_significant_params:
    print(f"Significant parameters found: {', '.join(significant_params)}")
    print(f"Iteration: {iterations}")
    random_data['cpu_freq'] = random_data['cpu_freq'].round(1)
    print(random_data)
    random_data.to_csv('random_data.csv', index=False)
