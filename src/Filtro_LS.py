"""Filtro LS"""

# Filtro LS Moore-Penrose

import numpy as np
from numpy.linalg import pinv as inversa
from Funcoes_auxiliares import matriz_observacao


# Versão 1 #
def filtro_LS(
    sinal_desejado: np.ndarray,
    readout: np.ndarray,
    ordem_filter: int,
):
    """
    1ª versão do Filtro LS para deconvolução do Circuito Shaper com o sinal de entrada.
    Método: Pseudo Inversa de Moore-Penrose
    """
    # Qantidade de leituras
    qntd_leitura = len(readout)

    # Matriz de Observação #
    matriz_obs = matriz_observacao(readout, ordem_filtro=ordem_filter)

    # Calculo peso #
    limite_filtro = qntd_leitura - ordem_filter + 3

    aux_peso = inversa(matriz_obs.T @ matriz_obs) @ matriz_obs.T
    peso = np.flipud(aux_peso @ sinal_desejado[2:limite_filtro])

    # Sinal Estimado/Recuperado #
    sinal_estimado = np.zeros(qntd_leitura)

    # Parte adaptativa do filtro
    len_sinal_estimado = qntd_leitura - ordem_filter + 1

    for i in range(len_sinal_estimado):
        sinal_estimado[i] = np.sum(readout[i : i + ordem_filter] * peso)

    return sinal_estimado


# Versão 2 #
def filtro_LS_2(  # filtro LS com delay, bias e clip #
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 2,  # <- novo: antes era [2:..], agr esse 2 é o delay e tem funçao pra ser calculado
    # bias,  # <- novo: equivalente ao bias de redes neurais
    # clipp: bool = True,  # <- novo: corta os valores abaixo de zero do sinal estimado e do readout
    valor_min_clip: int = 0,
    valor_max_clip: int | None = None,
):
    """
    2ª versão do Filtro LS para deconvolução do Shaper.
    Método: pseudo-inversa de Moore-Penrose.
    """

    # Substitui por 0 os valores abaixo de 0 da leitura
    readout_shaper = np.clip(readout, valor_min_clip, valor_max_clip)

    # Matriz de observação
    matriz_obs = matriz_observacao(readout_shaper, ordem_filtro=ordem_filter)

    # Tamanho útil para alinhar matriz_obs com y deslocado
    # end1 = len(readout_shaper) - ordem_filter + 3
    end = matriz_obs.shape[0] + delay

    sinal_desejado = np.asarray(sinal_desejado)

    # adiciona coluna de 1 para aprender o bias
    matriz_obs = np.column_stack([matriz_obs, np.ones(matriz_obs.shape[0])])

    # pesos via pseudo-inversa
    aux_peso = (
        inversa(matriz_obs.T @ matriz_obs) @ matriz_obs.T @ sinal_desejado[delay:end]
    )
    peso = np.flipud(aux_peso[:-1])
    bias = aux_peso[-1]

    # reconstrução/ Sinal Estimado/Recuperado #
    sinal_estimado = np.zeros(len(readout_shaper))
    max_i = len(readout_shaper) - ordem_filter + 1
    for i in range(max_i):
        # Parte adaptativa do filtro
        sinal_estimado[i] = np.sum(readout_shaper[i : i + ordem_filter] * peso) + bias

    # Substitui por 0 os valores abaixo de 0 da leitura
    sinal_estimado = np.clip(sinal_estimado, valor_min_clip, valor_max_clip)

    return sinal_estimado


# Versão 3 #
def filtro_LS_com_termos_nao_lineares(
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 0,
    # incluir_x2: bool = True,
    # incluir_xn_xn1: bool = True,
    valor_max_clip: int | None = None,
    valor_min_clip: int = 0,
):
    """
    LS com base expandida (não linear leve):
      - termos lineares x
      - termos quadráticos x^2
      - termos de interação x(n)*x(n-1)
    """
    readout_shaper = np.clip(readout, valor_min_clip, valor_max_clip)
    Xlin = matriz_observacao(readout_shaper, ordem_filtro=ordem_filter)
    y = np.asarray(sinal_desejado)[delay : delay + Xlin.shape[0]]

    # feats = [Xlin]
    # if incluir_x2:
    #     feats.append(Xlin**2)
    # if incluir_xn_xn1 and ordem_filter >= 2:
    #     feats.append(Xlin[:, 1:] * Xlin[:, :-1])

    feats = [Xlin, Xlin**2, Xlin[:, 1:] * Xlin[:, :-1]]

    Xexp = np.column_stack(feats)
    Xb = np.column_stack([Xexp, np.ones(Xexp.shape[0])])  # bias

    beta = inversa(Xb.T @ Xb) @ Xb.T @ y

    sinal_estimado = np.zeros(len(readout_shaper), dtype=float)
    sinal_estimado[: Xlin.shape[0]] = Xb @ beta
    sinal_estimado = np.clip(sinal_estimado, valor_min_clip, valor_max_clip)
    return sinal_estimado
