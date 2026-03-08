"""Filtro LMS"""

import numpy as np


def filtro_LMS0(
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


# Versão filtro LMS 1  #
def filtro_LMS1(
    sinal_desejado: np.ndarray,
    readout: np.ndarray,
    ordem_filter: int,
    delay: int = 2,
    mu: float = 1e-4,
    n_epocas: int = 10,
):
    """
    LMS baseado no filtro_LS1
    Retorna: sinal_estimado, peso, bias
    """
    # Quantidade de leituras
    qntd_leitura = len(readout)

    # Sinal estimado
    sinal_estimado = np.zeros(qntd_leitura)

    # Pesos e bias (inicialização)
    peso = np.zeros(ordem_filter, dtype=float)
    bias = 0.0

    # Parte adaptativa
    len_sinal_estimado = qntd_leitura - ordem_filter + 1
    # if len_sinal_estimado <= 0:
    #     return sinal_estimado, peso, bias

    d = np.asarray(sinal_desejado, dtype=float)
    x_in = np.asarray(readout, dtype=float)

    for _ in range(n_epocas):
        for i in range(len_sinal_estimado):
            idx_d = i + delay
            if idx_d >= len(d):
                break

            x = x_in[i : i + ordem_filter]
            y_hat = np.dot(x, peso) + bias
            erro = d[idx_d] - y_hat

            # Atualização LMS
            peso += mu * erro * x
            bias += mu * erro

            sinal_estimado[i] = y_hat

    # Mesmo padrão do LS1
    sinal_estimado = np.clip(sinal_estimado, 0, None)

    return sinal_estimado, peso, bias
