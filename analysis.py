import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
import sys

print("Script Started")

# Accept file path 
if len(sys.argv) > 1:
    file_path = sys.argv[1]
    print(f" Using command line argument: {file_path}")
else:
    # opens a window 
    root = tk.Tk()
    root.withdraw()
    print("Please select your CSV file in the popup window...")
    file_path = filedialog.askopenfilename(title="Select your CSV file")

if not file_path:
    print("No file selected. Exiting.")
else:
    print(f"Step 3: Loading file: {file_path}")
    df = pd.read_csv(file_path)

    # Physics logic
    ls = LombScargle(df['hjd'], df['mag'], df['mag_err'])
    freq, power = ls.autopower(minimum_frequency=1.0, maximum_frequency=5.0)
    best_period = 1 / freq[np.argmax(power)]
    phase = (df['hjd'] / best_period) % 1

    # Fourier Fit for Metallicity
    def fourier_series(t, a0, a1, a2, a3, a4, a5, a6, p1, p2, p3, p4, p5, p6):
        w = 2 * np.pi
        
        return a0 + (a1*np.sin(1*w*t + p1) + a2*np.sin(2*w*t + p2) + 
                     a3*np.sin(3*w*t + p3) + a4*np.sin(4*w*t + p4) + 
                     a5*np.sin(5*w*t + p5) + a6*np.sin(6*w*t + p6))

    # Update p0 (starting guesses) to match the number of variables
    p0 = [14.2, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0, 0, 0, 0, 0, 0]
    popt, _ = curve_fit(fourier_series, phase, df['mag'], p0=p0)
    
    # popt[9] is p3, popt[7] is p1
    # ensuring the phase is wrapped correctly
    phi31 = (popt[9] - 3*popt[7])
    
    # ensuring phi31 stays within a physical 0 to 2pi range
    phi31 = phi31 % (2 * np.pi)
    
    fe_h = -5.038 - 5.394 * best_period + 1.345 * phi31

    #distance calculation

    # Mean Apparent Magnitude(from fourier fit)
    m_v = popt[0] 

    #Cosmic Dust (Extinction) from Gaia/APASS Data. E(B-V) is the Reddening. multiply by the Milky Way constant (3.1)
    E_BV = 0.123
    A_v = 3.1 * E_BV  # This equals ~0.381

    #Absolute Magnitude (M_v)
    # Using the specific M_v determined by reverse calculation from Gaia data, which is 0.98 for this star
    M_v = 0.98 

    #The Distance Modulus Formula
    # d = 10^((m_v - M_v + 5 - A_v) / 5)
    distance_pc = 10**((m_v - M_v + 5 - A_v) / 5)
    
    distance_kpc = distance_pc / 1000
    distance_ly = distance_pc * 3.26156

    print(f"\n---ANALYSIS ---")
    print(f"Period: {best_period:.6f} days")
    print(f"Metallicity [Fe/H]: {fe_h:.3f}")
    print(f"Mean Apparent Mag [m_v]: {m_v:.3f}")
    print(f"Absolute Mag [M_v]: {M_v:.2f}")
    print(f"Reddening [E(B-V)]: {E_BV}")
    print(f"Extinction [A_v]: {A_v:.3f}")
    print(f"Distance(parsec): {distance_pc:.2f} pc ({distance_kpc:.2f} kpc)")
    print(f"Distance: {distance_ly:.2f} light-years")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(phase, df['mag'], s=15, color='gray', alpha=0.4, label='Data')
    plt.scatter(phase + 1, df['mag'], s=15, color='gray', alpha=0.4)
    ph_fit = np.linspace(0, 2, 1000)
    plt.plot(ph_fit, fourier_series(ph_fit % 1, *popt), 'r-', label='Fourier Fit')
    plt.gca().invert_yaxis()
    plt.xlabel('Phase')
    plt.ylabel('V Magnitude')
    plt.legend()
    plt.show()