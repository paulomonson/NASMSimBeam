import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
import math
 

class Graph:
    def __init__(self, title="Gráfico de Validação"):
        self.title = title
        self.beams = []
       
    def add_beam(self, beam):
        self.beams.append(beam)
    #grafico 1   
    def plot_beam_radius(self, z_max):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.beams)))
       
        for i, beam in enumerate(self.beams):
            z, w_z = beam.calculate_beam_radius(z_max)
            plt.plot(z, w_z, label=f'$w_0$ = {beam.w0*100:.2f} cm', color=colors[i])
            print(f"Para w0 = {beam.w0*100} cm, último w_z = {w_z[-1]:.6f} cm, z={z_max}m")
        
        
        plt.xlabel('Posição axial z (m)')
        plt.ylabel('Raio do feixe w_z (cm)')
        plt.title(f'{self.title} - Evolução do Raio do Feixe')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
       
    def plot_intensity_profile(self, z):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.beams)))
       
        for i, beam in enumerate(self.beams):
            intensity, x1, x2 = beam.calculate_intensity_profile(z)
            x = np.linspace(-beam.L/2, beam.L/2, beam.N)
            plt.plot(x, intensity, label=f'$w_0$ = {beam.w0*100:.2f} cm', color=colors[i])
           
            if x1 is not None:
                plt.axvline(x=x1, color=colors[i], linestyle='--', alpha=0.5)
                plt.axvline(x=x2, color=colors[i], linestyle='--', alpha=0.5)
       
        plt.axhline(y=1/np.exp(2), color='red', linestyle=':', label='1/e²')
        plt.xlabel('Posição transversal x (m)')
        plt.ylabel('Intensidade normalizada')
        plt.title(f'{self.title} - Perfil de Intensidade em z={z}m')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()