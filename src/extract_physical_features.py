# src/extract_physical_features.py
import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit

from .load_data import load_wave_file

def exp_decay(t, A, tau, C):
    """
    Одноэкспоненциальная модель затухающего сигнала:
        h(t) = A * exp(-t / tau) + C
    """
    return A * np.exp(-t / tau) + C

def extract_physical_features(path: str) -> pd.DataFrame:
    """
    Вычисляет физически осмысленные признаки для каждого сигнала.

    Признаки:
    - A   : амплитуда (максимальное значение сигнала);
    - S   : площадь под сигналом (интеграл по времени);
    - PSD : (long - short) / long, где
            long  — площадь всего сигнала,
            short — площадь окна сразу после максимума;
    - tau : эффективное время высвечивания, полученное из
            экспоненциальной аппроксимации хвоста сигнала.

    Параметры
    ---------
    path : str
        Путь к файлу с сигналами.

    Возвращает
    ----------
    df : pandas.DataFrame
        Таблица размера [N, 4] со столбцами A, S, PSD, tau.
    """
    # Загружаем данные на CPU (аппроксимация делается в NumPy/SciPy)
    meta, data = load_wave_file(path, device="cpu")
    x = data.numpy()  # [N, 500]
    N, T = x.shape

    # Гиперпараметры окна:
    # длина short-окна для PSD и длина хвоста для экспоненциального fit.
    short_len = 40
    tail_len = 150
    dt = 1.0  # шаг по времени в условных единицах

    A_list, S_list, PSD_list, tau_list = [], [], [], []

    for i in range(N):
        sig = x[i]

        # 1. Амплитуда и индекс максимума
        A = sig.max()
        imax = sig.argmax()

        # 2. Площадь под сигналом по всем отсчётам
        S = float(np.trapz(sig, dx=dt))

        # 3. PSD: нормированное соотношение "хвостовой" части к общей площади
        start_short = imax
        end_short = min(T, imax + short_len)
        short_area = float(np.trapz(sig[start_short:end_short], dx=dt))
        long_area = S
        PSD = (long_area - short_area) / long_area if long_area != 0 else 0.0

        # 4. Аппроксимация хвоста экспонентой для оценки tau
        start_tail = imax
        end_tail = min(T, imax + tail_len)
        t_tail = np.arange(end_tail - start_tail) * dt
        y_tail = sig[start_tail:end_tail]

        # Начальные приближения параметров экспоненты
        A0 = float(y_tail.max() - y_tail.min())
        tau0 = max((end_tail - start_tail) * dt / 3.0, 1e-3)
        C0 = float(y_tail.min())

        try:
            popt, _ = curve_fit(
                exp_decay, t_tail, y_tail,
                p0=[A0, tau0, C0],
                maxfev=2000
            )
            tau_val = float(abs(popt[1]))
        except Exception:
            # Если fit не сошёлся, оставляем пропуск (обработаем позже)
            tau_val = np.nan

        A_list.append(float(A))
        S_list.append(S)
        PSD_list.append(float(PSD))
        tau_list.append(tau_val)

    # Приведение tau к разумному диапазону и заполнение пропусков
    tau_arr = np.array(tau_list, dtype=np.float64)
    tau_min, tau_max = 0.0, 1e4

    bad_mask = (np.isnan(tau_arr)) | (tau_arr < tau_min) | (tau_arr > tau_max)
    good_tau = tau_arr[~bad_mask]
    tau_med = np.median(good_tau) if good_tau.size > 0 else 0.0
    tau_arr[bad_mask] = tau_med

    df = pd.DataFrame({
        "A": np.array(A_list, dtype=np.float32),
        "S": np.array(S_list, dtype=np.float32),
        "PSD": np.array(PSD_list, dtype=np.float32),
        "tau": tau_arr.astype(np.float32),
    })
    return df

if __name__ == "__main__":
    # Пример запуска: извлечение признаков из файла в папке data
    feats = extract_physical_features("data/Run200_Wave_0_1.txt")
    print(feats.head())
    print("Physical features shape:", feats.shape)
    feats.to_csv("physical_features.csv", index=False)
    print("Saved physical_features.csv")
