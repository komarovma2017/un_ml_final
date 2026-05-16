# src/load_data.py
import numpy as np
import torch
from typing import Tuple

def load_wave_file(path: str, device: str = "cpu") -> Tuple[np.ndarray, torch.Tensor]:
    """
    Загружает файл с сигналами формата N x 504.

    Первые 4 столбца считаются метаданными (возвращаются как NumPy),
    оставшиеся 500 столбцов — временные отсчёты сигналов (возвращаются как torch.Tensor).

    Параметры
    ---------
    path : str
        Путь к текстовому файлу с данными.
    device : str
        'cpu' или 'cuda', куда будет загружен тензор сигналов.

    Возвращает
    ----------
    meta : np.ndarray
        Массив размера [N, 4] с метаданными.
    data : torch.Tensor
        Тензор размера [N, 500] с сигналами.
    """
    raw = np.loadtxt(path)
    meta = raw[:, :4].astype(np.float32)
    signals = raw[:, 4:].astype(np.float32)

    # перевод в тензор сразу на нужное устройство
    data = torch.tensor(signals, device=device)

    return meta, data
