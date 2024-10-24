"""
Modelling Fickian diffusion of species in solution in a microfluidic device.

Microfluidic device consists of 3 inlets, each with a channel that joins
a larger main channel at the same point. Solutions are pumped through all 3
inlets at the same flow rate. 
        
This very simple model neglects parabolic velocity profile that forms in 
laminar flows, and the electrostatic or dispersive interactions between 
species on solution.

"""

import numpy as np


#%% MODELLING DIFFUSION

# Function to generate initial concentration distribution
def ini(x, uu, L, sigma):
    """
    sets the initial concentration distribution of the ions along the channel using a
    mathematical formula involving hyperbolic tangent functions.

    :param x: positions along channel width (in µm). array
    :param uu: initial concentration (in µM). float
    :param L: channel width (in µm). float
    :param sigma: sharpness of the transition in the concentration profile. int
    :return: rho: concentration distribution along the channel. array
    """
    rho = 0.5 * uu * (np.tanh(np.pi / sigma * (2 * L / 3 - x)) + 1) - 0.5 * uu * (np.tanh(np.pi / sigma * (L / 3 - x)) + 1)
    rho[rho < 1e-15] = 0
    return rho


# Diffusivity of each ion (in µm²/s)
DNa = 133.4
DCl = 203.2
DSO4 = 106.5

# Initial concentration (in µM) 
uT = 1e6

# Channel width (in µm)
L = 360

# Maximum time (in s)
tmax = 7

# FTCS setup
x = np.linspace(-L/2, L/2, 1001)
dx = np.abs(x[0]-x[1]) # step size
time = np.linspace(0, tmax, int(tmax/(100*10**(-6))))
dt = time[1] # time step

# Initial concentration distribution of salts
uNa = ini(x + L/2, uT, L, 8)
uCl = ini(x + L/2, uT, L, 8)
uSO4 = ini(x + L/2, uT, L, 8)

# Lists to store probability densities (ie concentration profiles at each time point)
probDensity_Na = []
probDensity_Cl = []
probDensity_SO4 = []

# Time points to run the algorithm
tpoints = [0, time[1000], time[2500], time[5000], time[7000], time[10000], time[14000], time[18000], time[22000], time[26000], time[30000], time[35000], time[40000]]

for uP, DP, probDensity in zip([uNa, uCl, uSO4], [DNa, DCl, DSO4], [probDensity_Na, probDensity_Cl, probDensity_SO4]):
    for t in time:
        uP[0] = uP[-1] = 0
        if t in tpoints:
            probDensity.append(uP.copy())
        uP[1:-1] += DP * (dt / dx**2) * (uP[2:] - 2 * uP[1:-1] + uP[:-2])
