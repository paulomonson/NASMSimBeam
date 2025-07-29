import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from math import ceil, log2, floor, log10
import math
 
class Feixe:
    def __init__(self, w0, wavelength=1550e-9, p=0, l=0, N=1024, L=None, tolerance=5e-3):
        self.w0 = w0
        self.wavelength = wavelength
        self.z_R = np.pi * w0**2 / wavelength
        self.p = p
        self.l = l
        self.N = N
        self.tolerance = tolerance
        self.L = L if L is not None else w0 *1000
        self.k = (2 * np.pi) / wavelength
        self.X, self.Y, self.R, self.Phi = self.calculate_grid()
 
    def calculate_grid(self):
        x = np.linspace(-self.L/2, self.L/2, self.N)
        y = np.linspace(-self.L/2, self.L/2, self.N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)
        return X, Y, R, Phi
 
    def laguerre_gauss_mode(self, z):
        w_z = self.w0 * np.sqrt(1 + (z / self.z_R)**2) * 100
        R_z = z * (1 + (self.z_R / z)**2) if z != 0 else np.inf
 
        norm = (np.sqrt(2 * (math.factorial(self.p)) / (np.pi * math.factorial(self.p + abs(self.l)))) * (1 / w_z))
        radial_term = (self.R * np.sqrt(2) / w_z)**abs(self.l)
        laguerre_poly = genlaguerre(self.p, abs(self.l))(2 * self.R**2 / w_z**2)
        gaussian = np.exp(-self.R**2 / w_z**2)
       
        phase = np.exp(-1j * (
            self.k * self.R**2 / (2 * R_z) +
            self.k * z +
            (2*self.p + abs(self.l) + 1) * np.arctan(z / self.z_R) -
            self.l * self.Phi
        ))
       
        E = norm * radial_term * laguerre_poly * gaussian * phase
        return E, w_z
 
    # Método para validação analítica
    def calculate_beam_radius(self, z_max, num_points=500):
        z_values = np.linspace(0, z_max, num_points)
        w_z_values = [self.w0 * np.sqrt(1 + (z / self.z_R)**2) * 100 for z in z_values]
        return z_values, w_z_values
 
    # Método para validação numérica
    def calculate_intensity_profile(self, z):
        E, _ = self.laguerre_gauss_mode(z)
        intensity = np.abs(E)**2
        central_line_index = self.N // 2
        intensity_profile = intensity[central_line_index, :]
        intensity_normalized = intensity_profile / np.max(intensity_profile)

        exact_matches = np.where(
            np.abs(intensity_normalized - 1/np.exp(2)) < self.tolerance
        )[0]
        x_matches = np.linspace(-self.L/2, self.L/2, self.N)[exact_matches]

        if len(x_matches) > 0:
            return intensity_normalized, x_matches[0], x_matches[-1]
        
        return intensity_normalized, x_matches[0], x_matches[-1]