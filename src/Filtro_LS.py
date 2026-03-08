"""Filtro LS"""

# Filtro LS Moore-Penrose

import numpy as np
from numpy.linalg import pinv as inversa
from Funcoes_auxiliares import matriz_observacao


# Versão 0 #
def filtro_LS0(
    sinal_desejado: np.ndarray,
    readout: np.ndarray,
    ordem_filter: int,
):
    """
    Versão 0 do Filtro LS para deconvolução do Circuito Shaper com o sinal de entrada.
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


# Versão 1 #
def filtro_LS1(
    sinal_desejado: np.ndarray,
    readout: np.ndarray,
    ordem_filter: int,
    delay: int = 2,
):
    """
    Versão 1 do Filtro LS para deconvolução do Circuito Shaper com o sinal de entrada.
    Método: Pseudo Inversa de Moore-Penrose
    """
    # Qantidade de leituras
    qntd_leitura = len(readout)

    # Matriz de Observação #
    matriz_obs = matriz_observacao(readout, ordem_filtro=ordem_filter)

    # Adiciona coluna de 1 para aprender o bias
    matriz_obs = np.column_stack([matriz_obs, np.ones(matriz_obs.shape[0])])

    # Calculo peso #
    limite_filtro = qntd_leitura - ordem_filter + 3

    # Pesos via pseudo-inversa
    aux_peso = (
        inversa(matriz_obs.T @ matriz_obs)
        @ matriz_obs.T
        @ sinal_desejado[delay:limite_filtro]
    )

    peso = np.flipud(aux_peso[:-1])  # -1 para retirar o bias
    bias = aux_peso[-1]

    # Sinal Estimado/Recuperado #
    sinal_estimado = np.zeros(qntd_leitura)

    # Parte adaptativa do filtro
    len_sinal_estimado = qntd_leitura - ordem_filter + 1

    for i in range(len_sinal_estimado):
        sinal_estimado[i] = np.dot(readout[i : i + ordem_filter], peso) + bias

    # Retira os valores menores que 0
    sinal_estimado = np.clip(sinal_estimado, 0, None)

    return sinal_estimado, peso, bias


# Versão 1.1 #
def filtro_LS10(
    sinal_desejado: np.ndarray,
    readout: np.ndarray,
    ordem_filter: int,
    delay: int = 2,
):
    """
    Versão 1 do Filtro LS para deconvolução do Circuito Shaper com o sinal de entrada.
    Método: Pseudo Inversa de Moore-Penrose
    """
    # Qantidade de leituras
    qntd_leitura = len(readout)

    # Matriz de Observação #
    matriz_obs = matriz_observacao(readout, ordem_filtro=ordem_filter)

    # Adiciona coluna de 1 para aprender o bias
    matriz_obs = np.column_stack([matriz_obs, np.ones(matriz_obs.shape[0])])

    # Calculo peso #
    limite_filtro = matriz_obs.shape[0]  # qntd_leitura - ordem_filter + 3

    # Pesos via pseudo-inversa
    aux_peso = (
        inversa(matriz_obs.T @ matriz_obs)
        @ matriz_obs.T
        @ sinal_desejado[delay : delay + limite_filtro]
    )

    peso = np.flipud(aux_peso[:-1])  # -1 para retirar o bias
    bias = aux_peso[-1]

    # Sinal Estimado/Recuperado #
    sinal_estimado = np.zeros(qntd_leitura)

    # Parte adaptativa do filtro
    len_sinal_estimado = qntd_leitura - ordem_filter + 1

    for i in range(len_sinal_estimado):
        sinal_estimado[i] = np.dot(readout[i : i + ordem_filter], peso) + bias

    # Retira os valores menores que 0
    sinal_estimado = np.clip(sinal_estimado, 0, None)

    return sinal_estimado, peso, bias


# Versão 2 # acho q esse cara ta bichado
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
    # end = len(readout_shaper) - ordem_filter + 3
    end = matriz_obs.shape[0] + delay

    sinal_desejado = np.asarray(sinal_desejado)

    # adiciona coluna de 1 para aprender o bias
    matriz_obs = np.column_stack([matriz_obs, np.ones(matriz_obs.shape[0])])

    # pesos via pseudo-inversa
    aux_peso = (
        inversa(matriz_obs.T @ matriz_obs) @ matriz_obs.T @ sinal_desejado[delay:end]
    )
    peso_com_bias = np.flipud(aux_peso)
    peso_sem_bias = peso_com_bias[:-1]
    bias = aux_peso[-1]

    # reconstrução/ Sinal Estimado/Recuperado #
    sinal_estimado = np.zeros(len(readout_shaper))
    max_i = len(readout_shaper) - ordem_filter + 1
    for i in range(max_i):
        # Parte adaptativa do filtro
        sinal_estimado[i] = (
            np.sum(readout_shaper[i : i + ordem_filter] * peso_sem_bias) + bias
        )

    # Substitui por 0 os valores abaixo de 0 da leitura
    sinal_estimado = np.clip(sinal_estimado, valor_min_clip, valor_max_clip)

    return sinal_estimado, peso_sem_bias, bias


# Versão 3 #
def filtro_LS_com_termos_nao_lineares(
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 0,
    incluir_x2: bool = True,
    incluir_xn_xn1: bool = True,
    valor_min_clip_entrada: np.floating | int | None = None,
    valor_max_clip_entrada: float | int | None = None,
    valor_min_clip_saida: np.floating | int | None = None,
    valor_max_clip_saida: float | int | None = None,
    retunr_pesos: bool = False,
):
    """
    Filtro LS (Least Squares) com expansão não-linear da base.

    Base utilizada:
        - Termos lineares: x
        - Termos quadráticos: x^2
        - Termos de interação: x(n) * x(n-1)

    Parâmetros:
    ------------

    sinal_desejado        : sinal alvo (target)
    readout               : sinal de entrada
    ordem_filter          : Ordem do filtro
    delay                 : atraso/delay
    incluir_x2            : Booleano para decidir se incluir o termo quadrático.
    incluir_xn_xn1        : Booleano para decidir se incluir o termo de interação entre amostras consecutivas
    valor_min_clip_entrada: valor mínimo para saturação do sinal de entrada. Para não clipar/retirar, None.
    valor_max_clip_entrada: valor máximo para saturação do sinal de entrada. Para não clipar/retirar, None.
    valor_min_clip_saida  : valor mínimo para saturação do sinal de saida. Para não clipar/retirar, None.
    valor_max_clip_saida  : valor máximo para saturação do sinal de saida. Para não clipar/retirar, None.
    retunr_pesos          : Booleano para decidir se os pesos também serão retornados.

    """

    # 1) PRÉ-PROCESSAMENTO DO SINAL DE ENTRADA
    readout = np.array(readout)

    # Garante que o sinal de entrada fique dentro do intervalo desejado
    readout_clipado = np.clip(
        readout, valor_min_clip_entrada, valor_max_clip_entrada, dtype=float
    )

    # 2) CONSTRUÇÃO DA MATRIZ DE OBSERVAÇÃO LINEAR

    # Cria matriz de observação
    X_linear = matriz_observacao(readout_clipado, ordem_filtro=ordem_filter)

    # Ajusta o sinal desejado considerando o delay
    s_desejado = np.asarray(sinal_desejado)[delay : delay + X_linear.shape[0]]

    # 3) EXPANSÃO NÃO-LINEAR DA BASE

    # 3.1 Termos lineares
    termos_lineares = X_linear
    feat = [termos_lineares]

    # 3.2 Termos quadráticos
    if incluir_x2:
        termos_quadraticos = X_linear**2
        feat.append(termos_quadraticos)

    # 3.3 Termos de interação entre amostras consecutivas
    #  x(n) * x(n-1)
    if incluir_xn_xn1:
        termos_interacao = X_linear[:, 1:] * X_linear[:, :-1]
        feat.append(termos_interacao)

    # Junta todas as features (colunas)
    X_expandida = np.column_stack(feat)

    # 4) ADIÇÃO DO TERMO DE BIAS

    # Cria coluna de 1's para representar o bias
    coluna_bias = np.ones((X_expandida.shape[0], 1))

    # Matriz final do modelo
    X_modelo = np.column_stack([X_expandida, coluna_bias])

    # 5) SOLUÇÃO DO PROBLEMA DE MÍNIMOS QUADRADOS

    # Fórmula pseudo_inversa para cálculo dos pesos
    inversa_matriz_normal = inversa(X_modelo.T @ X_modelo)
    pseudo_inversa = inversa_matriz_normal @ X_modelo.T

    pesos = pseudo_inversa @ s_desejado

    # 6) RECONSTRUÇÃO DO SINAL ESTIMADO

    sinal_estimado = np.zeros(len(readout_clipado), dtype=float)
    sinal_estimado[: X_linear.shape[0]] = X_modelo @ pesos

    # Aplica saturação no resultado final
    sinal_estimado = np.clip(sinal_estimado, valor_min_clip_saida, valor_max_clip_saida)

    # Retorno
    if not retunr_pesos:
        return sinal_estimado
    else:
        return sinal_estimado, pesos


# Versão 4 # em construção
def filtro_LS_nao_linear_adaptativo(
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 0,
    incluir_x2: bool = True,
    incluir_xn_xn1: bool = True,
    valor_min_clip_entrada: np.floating | int | None = None,
    valor_max_clip_entrada: float | int | None = None,
    valor_min_clip_saida: np.floating | int | None = None,
    valor_max_clip_saida: float | int | None = None,
    retunr_pesos: bool = False,
):
    """
    Filtro LS (Least Squares) com expansão não-linear da base.

    Base utilizada:
        - Termos lineares: x
        - Termos quadráticos: x^2
        - Termos de interação: x(n) * x(n-1)

    Parâmetros:
    ------------

    sinal_desejado        : sinal alvo (target)
    readout               : sinal de entrada
    ordem_filter          : Ordem do filtro
    delay                 : atraso/delay
    incluir_x2            : Booleano para decidir se incluir o termo quadrático.
    incluir_xn_xn1        : Booleano para decidir se incluir o termo de interação entre amostras consecutivas
    valor_min_clip_entrada: valor mínimo para saturação do sinal de entrada. Para não clipar/retirar, None.
    valor_max_clip_entrada: valor máximo para saturação do sinal de entrada. Para não clipar/retirar, None.
    valor_min_clip_saida  : valor mínimo para saturação do sinal de saida. Para não clipar/retirar, None.
    valor_max_clip_saida  : valor máximo para saturação do sinal de saida. Para não clipar/retirar, None.
    retunr_pesos          : Booleano para decidir se os pesos também serão retornados.

    """

    # 1) PRÉ-PROCESSAMENTO DO SINAL DE ENTRADA
    readout = np.array(readout)

    # Garante que o sinal de entrada fique dentro do intervalo desejado
    readout_clipado = np.clip(
        readout, valor_min_clip_entrada, valor_max_clip_entrada, dtype=float
    )

    # 2) CONSTRUÇÃO DA MATRIZ DE OBSERVAÇÃO LINEAR

    # Cria matriz de observação
    X_linear = matriz_observacao(readout_clipado, ordem_filtro=ordem_filter)

    # Ajusta o sinal desejado considerando o delay
    s_desejado = np.asarray(sinal_desejado)[delay : delay + X_linear.shape[0]]

    # 3) EXPANSÃO NÃO-LINEAR DA BASE

    # 3.1 Termos lineares
    termos_lineares = X_linear
    feat = [termos_lineares]

    # 3.2 Termos quadráticos
    if incluir_x2:
        termos_quadraticos = X_linear**2
        feat.append(termos_quadraticos)

    # 3.3 Termos de interação entre amostras consecutivas
    #  x(n) * x(n-1)
    if incluir_xn_xn1:
        termos_interacao = X_linear[:, 1:] * X_linear[:, :-1]
        feat.append(termos_interacao)

    # Junta todas as features (colunas)
    X_expandida = np.column_stack(feat)

    # 4) ADIÇÃO DO TERMO DE BIAS

    # Cria coluna de 1's para representar o bias
    coluna_bias = np.ones((X_expandida.shape[0], 1))

    # Matriz final do modelo
    X_modelo = np.column_stack([X_expandida, coluna_bias])

    # 5) SOLUÇÃO DO PROBLEMA DE MÍNIMOS QUADRADOS

    # Fórmula pseudo_inversa para cálculo dos pesos
    inversa_matriz_normal = inversa(X_modelo.T @ X_modelo)
    pseudo_inversa = inversa_matriz_normal @ X_modelo.T

    pesos = pseudo_inversa @ s_desejado

    # 6) RECONSTRUÇÃO DO SINAL ESTIMADO

    sinal_estimado = np.zeros(len(readout_clipado), dtype=float)
    sinal_estimado[: X_linear.shape[0]] = X_modelo @ pesos

    # Aplica saturação no resultado final
    sinal_estimado = np.clip(sinal_estimado, valor_min_clip_saida, valor_max_clip_saida)

    # Retorno
    if not retunr_pesos:
        return sinal_estimado
    else:
        return sinal_estimado, pesos


def filtro_LS_nao_linear_adaptativo2(
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 0,
    incluir_x2: bool = True,
    incluir_xn_xn1: bool = True,
    lambda_fator: float = 0.995,  # fator de esquecimento (0 < lambda <= 1)
    delta: float = 1e-2,  # regularização inicial de P
    valor_min_clip: float | int | None = None,
    valor_max_clip: float | int | None = None,
    retornar_pesos: bool = False,
):
    """
    LS não-linear adaptativo usando RLS (Recursive Least Squares) na base expandida.
    """

    if not (0 < lambda_fator <= 1):
        raise ValueError("lambda_fator deve estar em (0, 1].")
    if delta <= 0:
        raise ValueError("delta deve ser > 0.")

    readout_arr = np.asarray(readout, dtype=float)

    readout_clipado = np.clip(readout_arr, valor_min_clip, valor_max_clip)

    X_linear = matriz_observacao(readout_clipado, ordem_filtro=ordem_filter)
    sinal_desejado = np.asarray(sinal_desejado, dtype=float)

    n_amostras_validas = X_linear.shape[0]
    if delay + n_amostras_validas > len(sinal_desejado):
        raise ValueError("delay + amostras válidas excede o tamanho de sinal_desejado.")

    sinal_desejado = sinal_desejado[delay : delay + n_amostras_validas]

    # Define dimensão da base expandida
    dim = ordem_filter
    if incluir_x2:
        dim += ordem_filter
    if incluir_xn_xn1 and ordem_filter >= 2:
        dim += ordem_filter - 1
    # incluir_bias:
    dim += 1

    w = np.zeros(dim, dtype=float)
    P = (1.0 / delta) * np.eye(dim, dtype=float)

    y_hat = np.zeros(n_amostras_validas, dtype=float)

    for n in range(n_amostras_validas):
        x = X_linear[n]

        feats = [x]
        # incluir_x2:
        feats.append(x**2)
        # incluir_xn_xn1 and ordem_filter >= 2:
        feats.append(x[1:] * x[:-1])

        phi = np.concatenate(feats)
        # incluir_bias:
        phi = np.concatenate([phi, np.array([1.0])])

        # Predição com pesos atuais (online)
        y_hat[n] = float(w @ phi)
        erro = sinal_desejado[n] - y_hat[n]

        P_phi = P @ phi
        ganho = P_phi / (lambda_fator + float(phi @ P_phi))

        # Atualização adaptativa dos pesos (RLS)
        w = w + ganho * erro
        P = (P - np.outer(ganho, phi) @ P) / lambda_fator

    sinal_estimado = np.zeros(len(readout_clipado), dtype=float)
    sinal_estimado[:n_amostras_validas] = y_hat
    sinal_estimado = np.clip(sinal_estimado, valor_min_clip, valor_max_clip)

    if retornar_pesos:
        return sinal_estimado, w
    return sinal_estimado
