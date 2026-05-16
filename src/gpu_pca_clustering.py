# src/gpu_pca_clustering.py
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from .extract_physical_features import extract_physical_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def kmeans_torch(x: torch.Tensor, n_clusters: int = 3, n_iters: int = 100):
    """
    Реализация k-means на PyTorch с поддержкой GPU.

    Параметры
    ---------
    x : torch.Tensor
        Данные размера [N, D], находящиеся на CPU или CUDA.
    n_clusters : int
        Количество кластеров.
    n_iters : int
        Количество итераций алгоритма.

    Возвращает
    ----------
    labels : torch.Tensor
        Метки кластеров для каждой точки (shape [N]).
    centroids : torch.Tensor
        Координаты центроидов (shape [K, D]).
    inertia : float
        Сумма квадратичных расстояний до ближайших центроидов.
    """
    N, D = x.shape

    # Случайная инициализация центроидов по подмножеству точек
    indices = torch.randperm(N, device=x.device)[:n_clusters]
    centroids = x[indices].clone()

    for _ in range(n_iters):
        # Квадраты норм точек и центроидов для векторизованного подсчёта расстояний
        x2 = (x ** 2).sum(dim=1, keepdim=True)            # [N, 1]
        c2 = (centroids ** 2).sum(dim=1).unsqueeze(0)     # [1, K]
        distances = x2 + c2 - 2 * x @ centroids.T         # [N, K]

        # Назначаем каждую точку ближайшему центроиду
        labels = distances.argmin(dim=1)                  # [N]

        # Пересчитываем центроиды как средние по точкам в каждом кластере
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.any():
                centroids[k] = x[mask].mean(dim=0)

    # Финальный пересчёт inertia и меток
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    c2 = (centroids ** 2).sum(dim=1).unsqueeze(0)
    distances = x2 + c2 - 2 * x @ centroids.T
    labels = distances.argmin(dim=1)
    inertia = distances[torch.arange(N, device=x.device), labels].sum()

    return labels, centroids, float(inertia.item())

def main():
    # 1. Извлекаем физические признаки для всех сигналов
    feats_df = extract_physical_features("data/Run200_Wave_0_1.txt")  # [N, 4]
    feats = feats_df.values.astype(np.float32)

    # 2. Стандартизация признаков (ноль среднее, единичное отклонение)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-9
    feats_norm = (feats - mean) / std

    # 3. PCA до 2 компонент (сжатие в пространство максимальной дисперсии)
    pca = PCA(n_components=2, random_state=42)
    feats_pca = pca.fit_transform(feats_norm)  # [N, 2]

    # 4. Перенос в тензор на GPU (если доступен)
    x = torch.tensor(feats_pca, device=DEVICE)

    n_clusters = 3
    n_runs = 10
    n_iters = 100

    best_inertia = None
    best_labels = None

    # Несколько запусков k-means с разной инициализацией
    for run in range(n_runs):
        labels_gpu, centroids_gpu, inertia = kmeans_torch(
            x, n_clusters=n_clusters, n_iters=n_iters
        )
        print(f"run {run}: inertia = {inertia:.2f}")
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels_gpu.clone()

    # Переносим метки на CPU и сохраняем submission
    labels = best_labels.cpu().numpy()
    index = np.arange(labels.shape[0])
    df = pd.DataFrame({"index": index, "cluster": labels})
    df.to_csv("submission/submission.csv", index=False)
    print("Saved submission/submission.csv with shape:", df.shape, "best inertia:", best_inertia)

if __name__ == "__main__":
    main()
