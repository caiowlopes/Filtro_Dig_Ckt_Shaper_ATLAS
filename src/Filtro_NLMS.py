"""Filtro NLMS"""

import numpy as np


def filtro_NLMS(
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 2,
    mu: float = 0.5,
    epocas: int = 1,
    eps: float = 1e-8,
    valor_max_clip: int | float | None = None,
    valor_min_clip: int | float | None = 0,
):
    """
    NLMS: LMS normalizado (mais estavel quando energia de entrada varia).
    Retorna apenas sinal_estimado para ficar compativel com sua busca.
    """
    d = np.asarray(sinal_desejado, dtype=float)
    x_in = np.asarray(readout, dtype=float)

    x_in = np.clip(x_in, valor_min_clip, valor_max_clip)

    N = len(x_in)
    max_i = N - ordem_filter + 1
    if max_i <= 0:
        return np.zeros(N)

    w = np.zeros(ordem_filter, dtype=float)
    b = 0.0
    y_hat = np.zeros(N, dtype=float)

    for _ in range(epocas):
        for i in range(max_i):
            x = x_in[i : i + ordem_filter][::-1]
            idx_d = i + delay
            if idx_d >= len(d):
                break

            y = np.dot(w, x) + b
            e = d[idx_d] - y

            norm2 = np.dot(x, x)
            step = mu / (eps + norm2)

            w += step * e * x
            b += step * e

            y_hat[i] = y

    y_hat = np.clip(y_hat, valor_min_clip, valor_max_clip)
    return y_hat
