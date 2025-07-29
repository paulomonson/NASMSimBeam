import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
import math

class LaguerreGaussBeam:
    
    def __init__(self, w0, wavelength, p=0, l=0, N=1024, L=10.0):
        
        # Parâmetros fundamentais do feixe
        self.w0 = w0
        self.wavelength = wavelength
        self.p = p
        self.l = l
        
        self.N = N
        self.L = L
        
        self.k = 2 * np.pi / self.wavelength
        self.z_R = np.pi * self.w0**2 / self.wavelength
        
        x = np.linspace(-self.L / 2, self.L / 2, self.N)
        self.X, self.Y = np.meshgrid(x, x)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)
        print(f"Feixe inicializado com w₀ = {self.w0*100:.1f} cm. Distância de Rayleigh z_R = {self.z_R:.2f} m.")

    def laguerre_gauss_mode(self, z):
        
        # Evita divisão por zero no cálculo de R_z
        if abs(z) < 1e-9:
            R_z = np.inf
        else:
            R_z = z * (1 + (self.z_R / z)**2)

        # Raio do feixe em metros na distância z
        w_z_metros = self.w0 * np.sqrt(1 + (z / self.z_R)**2)
        
        # Termos da equação do campo de Laguerre-Gauss
        norm = (np.sqrt(2 * math.factorial(self.p) / (np.pi * math.factorial(self.p + abs(self.l)))) * (1 / w_z_metros))
        radial_term = (self.R * np.sqrt(2) / w_z_metros)**abs(self.l)
        laguerre_poly = genlaguerre(self.p, abs(self.l))(2 * self.R**2 / w_z_metros**2)
        gaussian_envelope = np.exp(-self.R**2 / w_z_metros**2)
        
        phase_term = np.exp(-1j * (
            self.k * self.R**2 / (2 * R_z) +
            self.k * z +
            (2 * self.p + abs(self.l) + 1) * np.arctan(z / self.z_R) -
            self.l * self.Phi
        ))
        
        E_field = norm * radial_term * laguerre_poly * gaussian_envelope * phase_term
        
        # Retorna o campo e o raio em centímetros
        return E_field, w_z_metros * 100

    def calculate_radius_analytical(self, z_values):
        """
        Calcula o raio do feixe usando a fórmula analítica.
        """
        w_z_analytical = self.w0 * np.sqrt(1 + (z_values / self.z_R)**2) * 100 # em cm
        return z_values, w_z_analytical

    def calculate_radius_numerical(self, z):
        """
        Calcula o raio do feixe numericamente a partir do perfil de intensidade 1/e^2.
        """
        E, _ = self.laguerre_gauss_mode(z)
        intensity = np.abs(E)**2
        
        # Pega um corte central do perfil de intensidade
        central_slice = intensity[self.N // 2, :]
        
        # Normaliza o perfil
        max_intensity = np.max(central_slice)
        if max_intensity < 1e-12:
            return None # Intensidade muito baixa para ser confiável
        
        normalized_intensity = central_slice / max_intensity
        
        # Coordenadas do eixo x em metros
        x_coords = np.linspace(-self.L / 2, self.L / 2, self.N)
        
        # Nível de intensidade alvo (1/e^2)
        target_level = 1 / np.e**2
        
        # Encontra os índices onde a intensidade cruza o nível alvo
        try:
            # Lado direito do centro
            indices_maiores_dir = np.where(normalized_intensity[self.N // 2:] > target_level)[0]
            idx_dir = indices_maiores_dir[-1] + self.N // 2
            
            # Lado esquerdo do centro
            indices_maiores_esq = np.where(normalized_intensity[:self.N // 2] > target_level)[0]
            idx_esq = indices_maiores_esq[0]
            
            # Interpolação linear para encontrar a posição exata
            # Lado direito
            y1, y2 = normalized_intensity[idx_dir], normalized_intensity[idx_dir + 1]
            x1, x2 = x_coords[idx_dir], x_coords[idx_dir + 1]
            pos_direita = x1 + (target_level - y1) * (x2 - x1) / (y2 - y1)
            
            # Lado esquerdo
            y1, y2 = normalized_intensity[idx_esq - 1], normalized_intensity[idx_esq]
            x1, x2 = x_coords[idx_esq - 1], x_coords[idx_esq]
            pos_esquerda = x1 + (target_level - y1) * (x2 - x1) / (y2 - y1)
            
            # O raio é metade da distância entre os dois pontos
            numerical_radius_m = (pos_direita - pos_esquerda) / 2
            return numerical_radius_m * 100 # Converte para cm
            
        except IndexError:
            # Ocorre se o feixe for maior que a grade de simulação
            print(f"Aviso: O feixe em z={z:.1f} m pode ser muito grande para a grade de simulação (L={self.L} m).")
            return None


if __name__ == '__main__':
    w0_values = [5e-2, 0.5e-2]
    wavelength = 1550e-9
    z_max = 10000
    num_numerical_points = 25
    
    # Criação da figura para o gráfico de validação
    fig_comp, axes_comp = plt.subplots(2, 1, figsize=(5, 10), constrained_layout=True)

    for i, w0 in enumerate(w0_values):
        ax_comp = axes_comp.flatten()[i]
        
        # Inicializa o feixe Gaussiano
        beam = LaguerreGaussBeam(w0=w0, wavelength=wavelength)
        
        # 1. Cálculo Analítico (curva contínua)
        z_analytical, w_analytical = beam.calculate_radius_analytical(np.linspace(1, z_max, 500))
        ax_comp.plot(z_analytical, w_analytical, 'b-', label='DFT')
        
        # 2. Cálculo Numérico (pontos discretos usando o método 1/e²)
        z_numerical = np.linspace(1, z_max, num_numerical_points)
        w_numerical = []
        valid_z = []
        for z in z_numerical:
            radius = beam.calculate_radius_numerical(z)
            if radius is not None:
                w_numerical.append(radius)
                valid_z.append(z)
        
        w_numerical = np.array(w_numerical)
        valid_z = np.array(valid_z)
        ax_comp.plot(valid_z, w_numerical, 'rx', markersize=8, markeredgewidth=2, label='Numerical (1/e²)')
        
        if len(w_numerical) > 0:
            w_final_ana = w_analytical[-1]
            w_final_num = w_numerical[-1]
            z_max_int = int(z_max)

            props = dict(boxstyle='round', pad=0.6, facecolor='wheat', alpha=0.75)
            
            line1 = f'DFT: {w_final_ana:.2f} cm'
            line2 = f'Numerical:  {w_final_num:.2f} cm'
            
            background_text = f'{line1}\n{line2}'
            
            ax_comp.text(0.05, 0.95, background_text, transform=ax_comp.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props, alpha=0)

            ax_comp.text(0.05, 0.95, line1, transform=ax_comp.transAxes, fontsize=10,
                       verticalalignment='top', color='blue')
            ax_comp.text(0.05, 0.90, line2, transform=ax_comp.transAxes, fontsize=10,
                       verticalalignment='top', color='red')
        # ---------------------------------------------------------

        # Configuração dos gráficos
        ax_comp.set_title(f'w₀ = {w0*100:.1f} cm', fontsize=14)
        ax_comp.set_xlabel('Propagation distance z (m)', fontsize=12)
        ax_comp.set_ylabel('Beam radius w(z) (cm)', fontsize=12)
        ax_comp.legend()
        ax_comp.grid(True, linestyle=':')

    plt.savefig('validation_laguerre_gauss.png', dpi=300)
    plt.show()