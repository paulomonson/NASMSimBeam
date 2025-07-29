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
w0 = 0.5e-2            # Raio da cintura do feixe
#w0=10e-2 
#w0=5e-2 
#w0=1e-2  
#w0=0.5e-2 

z_R = np.pi * w0**2 / wavelength  # Comprimento de Rayleigh
epsilon_0 = 8.854e-12   # Permissividade do vácuo
c = 3e8                 # Velocidade da luz
z = 0.6
#z_max = 10 
#z_max = 100  
#z_max = 1000 
#z_max = 5000 
#z_max = 10000 
  
 
k = (2*np.pi)/ wavelength
w_z = w0 * (np.sqrt(1 + (z / z_R)**2))  # Raio do feixe em z_max (~2.23 cm)
x0 = 0  
y0=0  
raio_fenda = 0.0004                       # Centro da gaussiana e da fenda


p, l = 0, 1            # Índices radial e azimutal (LGₚˡ)

#o grid deve ser mudado quase sempre a depender da distância z e do w0, 
#é no teste manual, sempre tendo em vista que quando menor o N mnos zoom na imagem e quando maior o L menos zoom na imagem 
#lembrando também que em dft não se pode tocar nas bordas pois produz efeitos não fisicos 
N= 1800
L= 0.02
   
M=N
dx = dy = L/N # Resolução espacial (0.5 µm/pixel)

# Grade de simulação ESSE É O GRID QUE TENHO QUE CONFIGURAR!!!!

x = np.arange(-N//2, N//2) * dx
y = np.arange(-N//2, N//2) * dy
X, Y = np.meshgrid(x, y)
r_squared = X**2 + Y**2
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

T2= f'{T}+ wave={wavelength} w0={w0*100}cm N={N} L={L} modo LG[{p},{l}]'

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

def feixe_nafenda(E0,X, Y, x0, y0):
    distancia_ao_centro = np.sqrt((X - x0)**2 + (Y - y0)**2)
    mascara_fenda= np.where(distancia_ao_centro <= raio_fenda, 1, 0)
    
    # Aplica a máscara à gaussiana
    E_nafenda = E0 * mascara_fenda
    return mascara_fenda, E_nafenda


def angular_spectrum_propagation(E0, E_nafenda, wavelength, dx, dy, z):
            
    # Coloca o campo inicial no epaço de fourier
    E0_fft = np.fft.fft2(E_nafenda)  
    
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

# Calcular intensidade em z=0
E0, _ = laguerre_gauss_mode(0)
#I_z0 =  np.abs(Ez0)**2  # Intensidade física em W/m²
I_z0 = (0.5 * epsilon_0 * c) * np.abs(E0)**2  # Intensidade física em W/m²

mascara_fenda, E_nafenda= feixe_nafenda(E0,X, Y, x0, y0)
       
plt.imshow(mascara_fenda)
plt.suptitle(f'fenda com raio = {raio_fenda*100} cm')
#plt.imshow(np.abs(E_fft_shifted)**2)
plt.show()

# Calcular intensidade em z=z_max
E0, _ = laguerre_gauss_mode(0)
E_zfinal = angular_spectrum_propagation(E0, E_nafenda, wavelength, dx, dy, z)
#I_zfinal =  np.abs(E_zfinal)**2  # Intensidade física em W/m²
I_zfinal = (0.5 * epsilon_0 * c) * np.abs(E_zfinal)**2  # Intensidade física em W/m²

# Criação da figura com 2 plots (1 linha, 2 colunas)
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1,3,width_ratios=[3,3,3])
plt.suptitle(f'Versão={T2}')

# Subplot 1: Mapa de calor em z=0
ax_mp1 = fig.add_subplot(gs[0,0])
heatmap1 = ax_mp1.imshow(np.abs(E_nafenda)**2, extent=[y.min(), y.max(), x.max(), x.min()], cmap='hot')
ax_mp1.set_title('Intensidade em z=0 (Origem)')
ax_mp1.set_xlabel('x (mm)')
ax_mp1.set_ylabel('y (mm)')
fig.colorbar(heatmap1, ax=ax_mp1, label='Intensity (W/m²)')

# Subplot 1: Mapa de calor em z=0
ax_mp1 = fig.add_subplot(gs[0,1])
heatmap1 = ax_mp1.imshow(I_z0, extent=[y.min(), y.max(), x.max(), x.min()], cmap='hot')
ax_mp1.set_title('Intensidade em z=0 (Origem)')
ax_mp1.set_xlabel('x (mm)')
ax_mp1.set_ylabel('y (mm)')
fig.colorbar(heatmap1, ax=ax_mp1, label='Intensity (W/m²)')

# Subplot 2: Mapa de calor em z=z_max
ax_mp2 = fig.add_subplot(gs[0,2])
heatmap2 = ax_mp2.imshow(I_zfinal, extent=[y.min(), y.max(), x.max(), x.min()], cmap='hot'  , vmin=0, vmax=0.016)
ax_mp2.set_title(f'Intensidade em z={z} m (Final)')
ax_mp2.set_xlabel('x (mm)')
ax_mp2.set_ylabel('y (mm)')
fig.colorbar(heatmap2, ax=ax_mp2, label='Intensity (W/m²)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
