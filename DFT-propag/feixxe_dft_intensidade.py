T='FEIXE COM DFT'
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import scipy
import math
from math import ceil, log2, floor, log10
import sys  # Adicione esta linha para usar sys.exit()
import matplotlib.gridspec as gridspec
from scipy.special import genlaguerre  # Polinômios de Laguerre generalizados

# Parâmetros do feixe
wavelength = 1550e-9  # Comprimento de onda (HeNe laser)
w0 = 5e-2            # Raio da cintura do feixe
#w0=10e-2 
#w0=5e-2 
#w0=1e-2  
#w0=0.5e-2 

z_R = np.pi * w0**2 / wavelength  # Comprimento de Rayleigh
epsilon_0 = 8.854e-12   # Permissividade do vácuo
c = 3e8                 # Velocidade da luz
z = 10000
#z_max = 10 
#z_max = 100  
#z_max = 1000 
#z_max = 5000 
#z_max = 10000 
 
k = (2*np.pi)/ wavelength

w_z = w0 * (np.sqrt(1 + (z / z_R)**2))  # Raio do feixe em z_max (~2.23 cm)


p, l = 0, 0            # Índices radial e azimutal (LGₚˡ)

#o grid deve ser mudado quase sempre a depender da distância z e do w0, 
#é no teste manual, sempre tendo em vista que quando menor o N mnos zoom na imagem e quando maior o L menos zoom na imagem 
#lembrando também que em dft não se pode tocar nas bordas pois produz efeitos não fisicos 
N=512
L= 0.9
   
M=N
dx = dy = L/N # Resolução espacial (0.5 µm/pixel)

# Grade de simulação ESSE É O GRID QUE TENHO QUE CONFIGURAR!!!!

x = np.arange(-N//2, N//2) * dx
y = np.arange(-N//2, N//2) * dy
X, Y = np.meshgrid(x, y, indexing='ij')
r_squared = X**2 + Y**2
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

T2= f'{T}+ wave={wavelength} w0={w0} N={N} L={L} modo LG[{p},{l}]'

# Função para calcular o campo LG
def laguerre_gauss_mode(z):
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)
    R_z = z * (1 + (z_R / z)**2) if z != 0 else np.inf

    # Termos da equação do campo de laguerre
    norm =(np.sqrt(2 * (math.factorial(p)) / (np.pi * math.factorial(p + abs(l)))) * (1 / w_z))
    radial_term = (R * np.sqrt(2) / w_z)**abs(l)
    laguerre_poly = genlaguerre(p, abs(l))(2 * R**2 / w_z**2)  # Avalia o polinômio nos pontos
    gaussian = np.exp(-R**2 / w_z**2)
    phase_1 = k * R**2 / (2 * R_z)
    phase_2= k*z
    phase_gouy = (2*p + abs(l) + 1) * np.arctan(z / z_R)
    phase_vortex = l * Phi
    phase0 = np.exp(-1j * (phase_1 + phase_2 + phase_gouy - phase_vortex))
    
    E0 = norm * radial_term * laguerre_poly * gaussian * phase0
    return E0, phase0

def angular_spectrum_propagation(E0, wavelength, dx, dy, z):
            
    # Coloca o campo inicial no epaço de fourier
    E0_fft = np.fft.fft2(E0)  
    
    Nx, Ny = E0.shape  # Use as dimensões reais do campo de entrada
    
    # Frequências espaciais (domínio de Fourier)
    fx = np.fft.fftfreq(Nx, dx)
    fy = np.fft.fftfreq(Ny, dy)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    
    # propagação (eq.paraxial)
    k_m_squared = (2 * np.pi * FX)**2 + (2 * np.pi * FY)**2
    H = np.exp(-1j * k_m_squared * z / (2 * k))
    
    E_z_fft = E0_fft * H
    
    # Transformada inversa para obter o campo propagado
    E_zfinal = np.fft.ifft2(E_z_fft)  
   
    return E_zfinal

# Criação da figura com 2 plots (1 linha, 2 colunas)
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1,2,height_ratios=[1], width_ratios=[3,3])
plt.suptitle(f'Versão={T2}')

# Calcular intensidade em z=0
Ez0, _ = laguerre_gauss_mode(0)
#I_z0 =  np.abs(Ez0)**2  # Intensidade física em W/m²
I_z0 = (0.5 * epsilon_0 * c) * np.abs(Ez0)**2  # Intensidade física em W/m²

# Calcular intensidade em z=z_max
E0, _ = laguerre_gauss_mode(0)
E_zfinal = angular_spectrum_propagation(E0, wavelength, dx, dy, z)
#I_zfinal =  np.abs(E_zfinal)**2  # Intensidade física em W/m²
I_zfinal = (0.5 * epsilon_0 * c) * np.abs(E_zfinal)**2  # Intensidade física em W/m²

# Subplot 1: Mapa de calor em z=0
ax_mp1 = fig.add_subplot(gs[0,0])
heatmap1 = ax_mp1.imshow(I_z0, extent=[y.min(), y.max(), x.max(), x.min()], cmap='hot', vmin=0, vmax=1)
ax_mp1.set_title('Intensidade em z=0 (Origem)')
ax_mp1.set_xlabel('x (m)')
ax_mp1.set_ylabel('y (m)')
fig.colorbar(heatmap1, ax=ax_mp1, label='Intensity (W/m²)', shrink = 0.8, location='left')

# Subplot 2: Mapa de calor em z=z_max
ax_mp2 = fig.add_subplot(gs[0,1])
heatmap2 = ax_mp2.imshow(I_zfinal, extent=[y.min(), y.max(), x.max(), x.min()], cmap='hot')
ax_mp2.set_title(f'Intensidade em z={z} m (Final)')
ax_mp2.set_xlabel('x (m)')
ax_mp2.set_ylabel('y (m)')
fig.colorbar(heatmap2, ax=ax_mp2, label='Intensidade (W/m²)',shrink = 0.8)

# Subplot 3: Gráfico 3D (fazer)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
