"""Filtro LMS"""

import numpy as np


def filtro_LMS(
    sinal_desejado,
    readout,
    ordem_filter=7,
    delay=2,
    mu=1e-4,
    n_epocas=1,
):
    y = np.asarray(readout, dtype=float)
    d = np.asarray(sinal_desejado, dtype=float)

    N = len(y)
    w = np.zeros(ordem_filter)
    b = 0.0
    s_est = np.zeros(N)

    max_i = N - ordem_filter + 1

    for _ in range(n_epocas):
        for i in range(max_i):
            x = y[i : i + ordem_filter][::-1]  # vetor de entrada
            dn = d[i + delay]  # alvo alinhado
            yn = np.dot(w, x) + b
            en = dn - yn

            w += mu * en * x
            b += mu * en

            s_est[i] = yn

    s_est = np.clip(s_est, 0, None)

    return s_est, w, b


# filtro_LMS 2  # talvez igual ao 1
def filtro_LMS2(
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 2,
    mu: float = 1e-4,
    epocas: int = 1,
    valor_max_clip: int | float | None = None,
    valor_min_clip: int | float | None = 0,
):
    """
    LMS (batch em epocas): adapta pesos amostra a amostra.
    Retorna apenas sinal_estimado para ficar compativel com sua busca.
    """
    d = np.asarray(sinal_desejado, dtype=float)
    x_in = np.asarray(readout, dtype=float)

    # opcional: clip no readout (mantendo seu padrao atual)
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
            x = x_in[i : i + ordem_filter][::-1]  # vetor de entrada
            idx_d = i + delay
            if idx_d >= len(d):
                break

            y = np.dot(w, x) + b
            e = d[idx_d] - y

            w += mu * e * x

            b += mu * e

            y_hat[i] = y

    y_hat = np.clip(y_hat, valor_min_clip, valor_max_clip)
    return y_hat
