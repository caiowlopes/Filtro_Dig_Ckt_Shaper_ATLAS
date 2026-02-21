"""Simulador de Filtros para Circuitos Shapers"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv as inversa
from typing import Callable
from src.Gerador_de_Sinal import main as gerador_sinal_entrada_saida

"""
Na minha primeira forma de buscar eu estava comparando os sinais original e estimado inclusive o pedaço que não era estimado.

Ambas as novas formas removem os zeros artificiais/não estimados

A 1 mudança: Cada ordem é avaliada no seu próprio tamanho válido (N - ordem + 1). Ordens diferentes usam janelas diferentes 


A 2 mudança: compara todas as ordens no mesmo intervalo temporal e na mesma quantidade de amostras.
"""
# Funções auxiliares #


# Plot comparação
def plot_estimado_x_original(
    original: np.ndarray,
    estimado: np.ndarray,
    limite_filtro: int | float,
    xlimite_min: int | float = 0,
    xlimite_max: int | float = 0,
    title: str = "Original x Estimado",
):
    # Plot
    plt.figure(figsize=(10, 6))

    if xlimite_max == 0:
        xlimite_max = 1.01 * len(original)

    plt.xlim(xlimite_min, xlimite_max)
    plt.title(title)
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")

    plt.plot(original)
    plt.plot(estimado)
    plt.axvline(limite_filtro, color="black", linestyle="--", linewidth=1.0)

    plt.legend(["Original", "Estimado", "Limite Estimativa"], loc="upper right")

    plt.grid()
    plt.show()


# Comparação numerica
def RMSE_e_MAE_por_ordem(A, B, printar: bool = False):
    """
    Calcula o RMSEW e o MAE. Ambos podem ser: np.array | int | floats...

    RMSE: root mean squared error
    MAE: mean absolute error

    Caso requisitado, os resultados são imprimidos.
    """
    diff = A - B

    rmse = np.sqrt(np.mean(diff**2))
    erro_abs_medio = np.mean(np.abs(diff))

    if printar:
        print(f"{erro_abs_medio = :.4f}")
        print(f"{rmse = :.4f}")

    return rmse, erro_abs_medio


# Função matriz de Observação
def matriz_observacao(sinal: list | np.ndarray, ordem_filtro: int = 2):
    """
    Constrói a matriz de observação a partir do sinal de entrada utilizando
    janelas deslizantes de tamanho igual à ordem do filtro.
    """
    return np.lib.stride_tricks.sliding_window_view(np.array(sinal), ordem_filtro)


# Funções do Filtro (LS)
# Filtro LS Moore-Penrose
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


# Filtro LS Moore-Penrose com alterações/novidades
def filtro_LS_novo(  # filtro LS com delay, bias e clip #
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 2,  # <- novo: antes era [2:..], agr esse 2 é o delay e tem funçao pra ser calculado
    # bias,  # <- novo: equivalente ao bias de redes neurais
    # clipp: bool = True,  # <- novo: corta os valores abaixo de zero do sinal estimado e do readout
    valor_max_clip: int | None = None,
    valor_min_clip: int = 0,
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


# Busca melhor ordem do filtro #
# Busca considerando janelas diferentes, cada ordem com a sua
def busca_ordem_otima_filtro(
    signal_original: np.ndarray,
    ordem_mais_alta: int = 10,
    step: int = 1,
    filtro: Callable = filtro_LS,
    tamanho_janela_fixo=False,
    delay: int = 0,
    **filtro_kwargs,
):
    """
    Esse filtro não aceita ordem menor do que 3.

    Parametros:
    -----------
    ordem_mais_alta: int, obrigatorio.
         valor da ordem mais alta a ser testada.
    """

    melhor_ordem_dict = {
        "Ordem_Filtro_RMSE": 0,
        "RMSE": np.inf,
        "Ordem_Filtro_MAE": 0,
        "MAE": np.inf,
    }

    if ordem_mais_alta < 3:
        ordem_mais_alta = 3

    for ordem_filter in range(3, ordem_mais_alta + 1, step):

        sinal = filtro(
            sinal_desejado=signal_original,
            ordem_filter=ordem_filter,
            **filtro_kwargs,
        )

        if tamanho_janela_fixo:
            janela = len(signal_original) - ordem_filter + 1
            # estimado_eval = sinal[:janela]
            # original_eval = signal_original[delay : delay + janela]
        else:  # considerando a mesma janela para todos
            # Janela comum (fixa) para todas as ordens candidatas
            # Mesmo trecho para todas as ordens
            janela = len(signal_original) - ordem_mais_alta + 1

        if janela <= 0:
            raise ValueError(
                "Sinal curto demais para a ordem_mais_alta informada. "
                "É necessário len(signal_original) - ordem_mais_alta + 1 > 0."
            )

        estimado_eval = sinal[:janela]
        original_eval = signal_original[delay : delay + janela]

        rmse, mae = RMSE_e_MAE_por_ordem(estimado_eval, original_eval)

        if rmse < melhor_ordem_dict["RMSE"]:
            melhor_ordem_dict["RMSE"] = round(rmse, 10)
            melhor_ordem_dict["Ordem_Filtro_RMSE"] = ordem_filter

        if mae < melhor_ordem_dict["MAE"]:
            melhor_ordem_dict["MAE"] = round(mae, 10)
            melhor_ordem_dict["Ordem_Filtro_MAE"] = ordem_filter

    return melhor_ordem_dict


# Busca melhor delay do filtro #
def busca_delay_otimo(
    delay_maximo: int,
    signal_original: np.ndarray,
    filtro: Callable = filtro_LS,
    **filtro_kwargs,
):
    melhor = {
        "Delay_RMSE": None,
        "RMSE": np.inf,
        "Delay_MAE": None,
        "MAE": np.inf,
    }

    for delay in range(delay_maximo):
        sinal = filtro(
            sinal_desejado=signal_original,
            delay=delay,
            **filtro_kwargs,
        )
        rmse, mae = RMSE_e_MAE_por_ordem(sinal, signal_original)

        if rmse < melhor["RMSE"]:
            melhor["RMSE"] = rmse
            melhor["Delay_RMSE"] = delay

        if mae < melhor["MAE"]:
            melhor["MAE"] = mae
            melhor["Delay_MAE"] = delay

    melhor["RMSE"] = round(float(melhor["RMSE"]), 5)
    melhor["MAE"] = round(float(melhor["MAE"]), 5)
    return melhor


# Consantes & Variáveis #

quantidade_de_amostras = 75

sinal_original, Readout_Shaper = gerador_sinal_entrada_saida(
    quantidade_de_amostras, seed=42
)


# Filtro LS "antigo"

melhor_ordem = busca_ordem_otima_filtro(
    ordem_mais_alta=21,
    signal_original=sinal_original,
    readout=Readout_Shaper,
    filtro=filtro_LS,
)

RMSE = melhor_ordem["RMSE"]
MAE = melhor_ordem["MAE"]
ordem_f_rmse = melhor_ordem["Ordem_Filtro_RMSE"]
ordem_f_mae = melhor_ordem["Ordem_Filtro_MAE"]

print(f"Melhor ordem achada baseado em RMSE: {ordem_f_rmse}")
print(f"Melhor ordem achada baseado em MAE: {ordem_f_mae}\n")

print(f"RMSE do filtro antigo de ordem {ordem_f_rmse}: {RMSE}")
print(f"MAE do filtro antigo de ordem {ordem_f_rmse}: {MAE}\n")

sinal_estimado = filtro_LS(
    readout=Readout_Shaper,
    sinal_desejado=sinal_original,
    ordem_filter=ordem_f_rmse,
)

rmse_LS1, _ = RMSE_e_MAE_por_ordem(sinal_estimado, sinal_original, True)

plot_estimado_x_original(
    estimado=sinal_estimado,
    original=sinal_original,
    limite_filtro=len(Readout_Shaper) - ordem_f_rmse + 1,
    title="Original x Estimado antigo",
)


# filtro_old()
# Transiçao #
print("--------------------------------------------------------")


# Filtro LS "novo" #

melhor_ordem_f2 = busca_ordem_otima_filtro(
    ordem_mais_alta=21,
    signal_original=sinal_original,
    readout=Readout_Shaper,
    filtro=filtro_LS_novo,
)

RMSE_2 = melhor_ordem_f2["RMSE"]
MAE_2 = melhor_ordem_f2["MAE"]
ordem_f_rmse2 = melhor_ordem_f2["Ordem_Filtro_RMSE"]
ordem_f_mae2 = melhor_ordem_f2["Ordem_Filtro_MAE"]

print(f"Melhor ordem achada baseado em RMSE: {ordem_f_rmse2}")
print(f"Melhor ordem achada baseado em MAE: {ordem_f_mae2}\n")

print(f"RMSE do filtro novo de ordem {ordem_f_rmse2}: {RMSE_2}")
print(f"MAE do filtro novo de ordem {ordem_f_mae2}: {MAE_2}\n")

melhor_delay = busca_delay_otimo(
    delay_maximo=7,
    signal_original=sinal_original,
    filtro=filtro_LS_novo,
    readout=Readout_Shaper,
    ordem_filter=ordem_f_rmse2,
)

best_delay_2 = melhor_delay["Delay_RMSE"]

RMSE_2_delay = melhor_delay["RMSE"]
MAE_2_delay = melhor_delay["MAE"]

print(f"Melhor delay achada baseado em RMSE: {best_delay_2}")
print(f"Melhor delay achada baseado em MAE: {melhor_delay['Delay_MAE']}")

print(f"RMSE do melhor delay {best_delay_2}: {RMSE_2_delay}")
print(f"MAE do melhor delay {best_delay_2}: {MAE_2_delay}\n")

sinal_estimado_2 = filtro_LS_novo(
    readout=Readout_Shaper,
    sinal_desejado=sinal_original,
    ordem_filter=ordem_f_rmse2,
    delay=best_delay_2,
)

rmse_LS2, _ = RMSE_e_MAE_por_ordem(sinal_estimado_2, sinal_original, True)

plot_estimado_x_original(
    estimado=sinal_estimado_2,
    original=sinal_original,
    limite_filtro=len(Readout_Shaper) - ordem_f_rmse2 + 1,
    title="Original x Estimado novo",
)

# Para varios run:
# for i in range(10):
#     sinal_original, Readout_Shaper = gerador_sinal_entrada_saida(
#     quantidade_de_amostras, seed=42+i
# )
#     filtro_LS()
#     filtro_LS_novo()
