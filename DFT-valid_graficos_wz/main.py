from feixgaussiano import Feixe
import matplotlib.pyplot as plt
import numpy as np

# Valores de w0 em cm (convertidos para metros)
w0_values = [10e-2, 5e-2, 1e-2, 0.5e-2]
# Valores de z_max em metros (adicione mais se quiser comparar mais distâncias)
z_max_values = [1000]  # pode adicionar outros: [10, 100, 1000, 5000, 10000]

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(w0_values) * len(z_max_values)))

idx = 0
for z_max in z_max_values:
    for w0 in w0_values:
        feixe = Feixe(w0=w0, N=1024, L=8)
        z, w_z = feixe.calculate_beam_radius(z_max)
        plt.plot(z, w_z, label=f'$w_0$={w0*100:.2f}cm, $z_{{max}}$={z_max}m, w_z={w_z[-1]:.3f}', color=colors[idx])

        idx += 1


# grafico = Graph("Perfil de Intensidade")
# grafico.add_beam(feixe)
# grafico.plot_intensity_profile(z=z_max)

plt.xlabel('Posição axial z (m)')
plt.ylabel('Raio do feixe $w_z$ (cm)')
plt.title('Evolução do Raio do Feixe para diferentes $w_0$ e $z_{max}$')
plt.legend()
plt.grid()
plt.tight_layout()



# Gráfico com subplots: cada cintura (w0) em um gráfico, todos na mesma imagem
fig, axs = plt.subplots(1, len(w0_values), figsize=(5*len(w0_values), 5), sharey=True)
if len(w0_values) == 1:
    axs = [axs]
for i, w0 in enumerate(w0_values):
    feixe = Feixe(w0=w0, N=512, L= 40)  # L aumentado para 40
    z = z_max_values[0]
    intensity, x1, x2 = feixe.calculate_intensity_profile(z=z)
    x = np.linspace(-feixe.L/2, feixe.L/2, feixe.N)
    axs[i].plot(x, intensity, color='blue')
    if x1 is not None and x2 is not None:
        axs[i].axvline(x=x1, color='green', linestyle='--', alpha=0.5)
        axs[i].axvline(x=x2, color='green', linestyle='--', alpha=0.5)
        wz_val = abs(x2 - x1)/2
        wz_str = f"w_z = {wz_val:.4f} m"
    else:
        wz_str = "w_z = N/A"
    axs[i].axhline(y=1/np.exp(2), color='red', linestyle=':', label='1/e²')
    axs[i].set_xlabel('Posição transversal x (m)')
    axs[i].set_title(f'$w_0$={w0*100:.2f}cm\n z={z}m\n{wz_str}')
    axs[i].grid()
    axs[i].legend(['Perfil', '1/e²'])
axs[0].set_ylabel('Intensidade normalizada')
fig.suptitle(f'Perfis de Intensidade em z={z_max_values[0]}m para diferentes $w_0$', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()