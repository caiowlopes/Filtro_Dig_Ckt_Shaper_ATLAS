# Bibliotecas #
from Filtro_LS import *
from Funcoes_auxiliares import *
from Gerador_de_Sinais import main as gerador_sinal_entrada_saida


# Consantes & Variáveis #
quantidade_de_amostras = 1000
dt = 25 * 10**-9  # tempo entre amostras. Dados proveniente do Shaper
fs = 1 / dt  # frequência de amostragem. 40 MHz
resoluçao = fs / quantidade_de_amostras

# Gerador de sinais #
sinal_original, Readout_Shaper = gerador_sinal_entrada_saida(
    quantidade_de_amostras, seed=42
)


def zona_maior_erro(xh, x=sinal_original, limite=quantidade_de_amostras - 7 + 1):
    """
    x: sinal real
    xh: sinal estimado
    """
    erro = x[:limite] - xh[:limite]
    erro_abs = np.abs(erro)

    idx_max = np.argmax(erro_abs)  # posição do maior erro
    maior_erro = erro[idx_max]  # erro com sinal
    maior_erro_abs = erro_abs[idx_max]  # magnitude do erro

    print(f"Maior erro em i={idx_max}")
    print(f"erro = {maior_erro:.6f}")
    print(f"|erro| = {maior_erro_abs:.6f}")
    print(f"x_real = {x[idx_max]:.6f}, x_est = {xh[idx_max]:.6f}")

    return maior_erro_abs, xh[idx_max]


# Plot Original x filtro_LS_1 #

ordem_f_LS1 = 7

s_est_LS1 = filtro_LS(
    sinal_desejado=sinal_original,
    readout=Readout_Shaper,
    ordem_filter=ordem_f_LS1,
)

lim_filt1 = quantidade_de_amostras + 1 - ordem_f_LS1

r_1, m_1 = RMSE_e_MAE_por_ordem(
    A=s_est_LS1,
    B=sinal_original,
    limite_filtro=lim_filt1,
    printar=True,
)

plot_estimado_x_original(
    sinal_original,
    estimado=s_est_LS1,
    limite_filtro=quantidade_de_amostras - ordem_f_LS1 + 1,
    title="Original x filtro_LS_1",
)

# Plot Original x filtro_LS_2 #

ordem_f_LS2 = 7

s_est_LS2 = filtro_LS_2(
    sinal_desejado=sinal_original,
    readout=Readout_Shaper,
    ordem_filter=ordem_f_LS2,
)

lim_filt2 = quantidade_de_amostras + 1 - ordem_f_LS2

r_2, m_2 = RMSE_e_MAE_por_ordem(
    A=s_est_LS2,
    B=sinal_original,
    limite_filtro=lim_filt2,
    printar=True,
)


plot_estimado_x_original(
    sinal_original,
    estimado=s_est_LS2,
    limite_filtro=quantidade_de_amostras - ordem_f_LS2 + 1,
    title="Original x filtro_LS_2",
)

# Plot Original x filtro_LS_com_termos_nao_lineares #
ordem_f_LS_ext = 7

s_est_LS_ext = filtro_LS_com_termos_nao_lineares(
    sinal_desejado=sinal_original,
    readout=Readout_Shaper,
    ordem_filter=ordem_f_LS_ext,
    delay=0,
)

lim_filt_ext = quantidade_de_amostras + 1 - ordem_f_LS_ext


print(f"{quantidade_de_amostras = }")


lim_filt_ext = quantidade_de_amostras - ordem_f_LS_ext + 1

r_ext, m_ext = RMSE_e_MAE_por_ordem(
    A=s_est_LS_ext,
    B=sinal_original,
    limite_filtro=lim_filt_ext,
    printar=True,
)

plot_estimado_x_original(
    sinal_original,
    estimado=s_est_LS_ext,
    limite_filtro=quantidade_de_amostras - ordem_f_LS_ext + 1,
    title="Original x filtro_LS_com_termos_nao_lineares",
)
limite = quantidade_de_amostras - 7 + 1

x = sinal_original[:limite]
xh = s_est_LS_ext[:limite]

maior_erro_abs, y_pico = zona_maior_erro(s_est_LS_ext)
erro = x - xh
erro_abs = np.abs(erro)
idx_max = np.argmax(erro_abs)

janela = 80  # pontos para cada lado
ini = max(0, idx_max - janela)
fim = min(len(x), idx_max + janela + 1)

# eixo x do plot

eixo = np.arange(ini, fim)
x_label = "Amostra"
x_pico = float(idx_max)

plt.figure(figsize=(10, 4))
plt.plot(eixo, x[ini:fim], label="Sinal real", linewidth=2)
plt.plot(eixo, xh[ini:fim], label="Estimativa", linewidth=2, alpha=0.85)
plt.axhline(y=y_pico, color="r", linestyle=":", label="Maior erro")
plt.axvline(x=x_pico, color="r", linestyle=":")
plt.title(f"Região do maior erro (idx={idx_max}, |erro|={erro_abs[idx_max]:.4f})")
plt.xlabel(x_label)
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
