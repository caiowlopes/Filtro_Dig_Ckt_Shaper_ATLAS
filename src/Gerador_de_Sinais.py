# Bibliotecas

import numpy as np
import Shaper_ATLAS_Simulator as Shaper_ATLAS_Simulator
import matplotlib.pyplot as plt
from typing import Callable

# import src.Shaper_ATLAS_Simulator as Shaper_ATLAS_Simulator


# Funções auxiliares #


# Função sinal leitura (saida shaper) #
# sinal_original = trem de impulsos # energy truth (sinal de enregia verdadeiro)
def simulate_shaper_readout(
    lista: list | np.ndarray,
    matriz: list | np.ndarray,
    return_resul1: bool = False,
):
    """
    Saida esperada:

    entrada:
    lista = np.array([1, 2, 3, 4])
    matriz = np.array(
        [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]]
        )

    saida:
    resultado1 = array( # a11 da lista x linha 1 da matriz. a12 da lista vezes linha 3 da matriz deslocado
        [[11, 12, 13, 14],
        [ 0, 30, 32, 34],
        [ 0,  0, 57, 60],
        [ 0,  0,  0, 92]
        ])

    resultado2 = array([ 11,  42, 102, 200])) # soma das colunas
    """

    # n_linhas, n_cols = np.array(matriz).shape
    n_linhas = len(matriz)
    n_cols = len(matriz[0])

    # resultado1: matriz deslocada e multiplicada
    resultado1 = np.zeros((n_linhas, n_cols))
    for linha in range(n_linhas):
        for cols in range(n_cols - linha):
            resultado1[linha][linha + cols] = lista[linha] * matriz[linha][cols]

    resultado1 = np.array(resultado1)

    # resultado2: soma por coluna
    resultado2 = resultado1.sum(axis=0)

    if return_resul1:
        return resultado1, resultado2
    else:
        return resultado2


# Gerador/Simulador do sinal registrado pelos sensores #
def original_signal_generator(
    num_amostras_leitura: int,
    position_percentage: float = 0.2,
    media_energia_cintilador: int = 30,
    seed=None,
):
    """
    position_percentage = 20 a 30 % de num_amostras_leitura
    """
    rng = np.random.default_rng(seed)

    signal = np.zeros(num_amostras_leitura)

    # Sortear indices de signal
    position = int(position_percentage * num_amostras_leitura)

    indx = rng.permutation(len(signal))[:position]
    signal[indx] = rng.exponential(media_energia_cintilador, size=len(indx)).astype(int)

    "Forma antes d mudar para usar o seed"
    # As posições
    # indx = np.random.permutation(len(signal))[:position]
    # As posições com as devidas energias
    # signal[indx] = np.random.exponential(
    #     media_energia_cintilador, size=len(indx)
    # ).astype(int)

    return signal


# Gerando Formas de onda em diversos cenários #
def matriz_convolucao(
    amostras_das_leitura: int,
    CKT_parameters: Callable | None,
    noise: int | float | list | np.ndarray = 0,
    CKT_simulator: Callable | None = None,
    parameters: dict | None = None,
    Seed: int | None = None,
):
    """
    Função para gerar a matriz de convolução de um circuito.
    A matriz resultate é qudrada (amostras_leitura x amostras_leitura).


    Parametros:
    -----------
    amostras_leitura: quantidade de amostras na leitura

    noise: erro/ruído associado a leitura do sinal para cada elemento do circuito. No caso d ser lista/array, deve ter o mesmo len() que Componentes, de CKT_parameters.

    CKT_parameters: parametros do circuito a ser simulado. Deve, obrigatoriamente, retornar 3 objetos/variaveis, nesta ordem:
        1. Função de transferência do circuito com as variaveis ainda a serem substituidas (funçao literal com o sympy),
        2. Componentes: Lista de variaveis do circuito como objeto sympy.symbols
        3. Valor_componentes: Lista dos valores dos componentes do circuito respeitando a ordem da variavel Componentes.

    parameters: Dicionario com as variáveis a serem desempacotadas para a função CKT_simulator().

    CKT_simulator: Função que faz a simulção do circuito.

    Return:
    -------
    Retorna os resultados da simulação do circuito de CKT_simulator().
    """

    # Parametros do CKT #

    TF, component, component_values = CKT_parameters()  # type: ignore[operator]

    # Parametros do gerador das formas de onda #
    if parameters is None:
        parametros_MC = {
            "iterations": amostras_das_leitura,
            "t": np.arange(0, amostras_das_leitura) * 25 * 10**-9,
            "FT": TF,  # literal
            "erro": noise,  # Erro de cada elemento do circuito
            "components": component,
            "nominal_values": component_values,
            "seed": Seed,
        }

    return CKT_simulator(**parametros_MC)  # type: ignore[operator]


def main(
    sinal_original: np.ndarray,
    ckt_parameters_error: np.ndarray | None = None,
    CKT_parameters: Callable = Shaper_ATLAS_Simulator.ckt_parameters,
    CKT_simulator: Callable = Shaper_ATLAS_Simulator.MonteCarlo_iteration,
    seed=None,
):
    """
    sinal_original: sinal proveniente das colisoes. Sinal desejado.
    seed: para reprodutibilidade da aleatoriedade do Monte Carlo.
    "CKT_simulator": Script_TCC.MonteCarlo_iteration,
    "CKT_parameters": Script_TCC.ckt_parameters,
    "ckt_parameters_error": ckt_parameters_error,
    """

    if ckt_parameters_error is None:
        ckt_parameters_error = (  # dois ultimos são tau1 e tau2
            np.array([10, 1, 1, 1, 2, 2, 2, 0.10, 0, 0], dtype=float)
        ) / 100

    # Simulação do efeito do Shaper
    wave_former = np.asarray(
        matriz_convolucao(
            amostras_das_leitura=len(sinal_original),
            noise=ckt_parameters_error,
            CKT_parameters=CKT_parameters,
            CKT_simulator=CKT_simulator,
            Seed=seed,
        )[-1]
    )

    # Sinal após passar pelo Shaper
    Readout_Shaper = simulate_shaper_readout(matriz=wave_former, lista=sinal_original)

    return Readout_Shaper


if __name__ == "__main__":
    # Parametros pré-filtro #

    # Constantes #
    media_energia_cada_cintilador = 30

    # Variaveis #
    qntd_amostras_leitura = 50
    ckt_parameters_error = (  # dois ultimos são tau1 e tau2
        np.array([10, 1, 1, 1, 2, 2, 2, 0.10, 0, 0], dtype=float)
    ) / 100

    s_desejado = original_signal_generator(qntd_amostras_leitura)

    parametro_leitura_ckt = {
        "CKT_parameters": Shaper_ATLAS_Simulator.ckt_parameters,
        "CKT_simulator": Shaper_ATLAS_Simulator.MonteCarlo_iteration,
        "ckt_parameters_error": ckt_parameters_error,
        "sinal_original": s_desejado,
        # "media_energia_cada_cintilador": 30,
    }

    leitura_do_Shaper = main(**parametro_leitura_ckt)

    plt.plot(leitura_do_Shaper)
    plt.grid()
    plt.show()

    print("ok")
