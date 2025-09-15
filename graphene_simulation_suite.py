# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:52:02 2024

@author: mt840
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:09:12 2024

@author: mt840
"""

import importlib.util
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import make_interp_spline


class GrapheneSimulation:
    def __init__(self, simulation_type, mu_values, gamma_values, slg_overlap):
        self.simulation_type = simulation_type
        self.mu_values = mu_values
        self.gamma_values = gamma_values
        self.slg_overlap = slg_overlap
        self.lumapi_module = self.load_lumapi_module()
        
    def load_lumapi_module(self):
        #lumapi_path = "C:\\Program Files\\Lumerical\\v221\\api\\python\\lumapi.py"  # Mat laptop
        #lumapi_path = "C:\\Program Files\\Lumerical\\v222\\api\\python\\lumapi.py"  # CGC desktop
        lumapi_path = "C:\Program Files\Lumerical\v251\api\python\lumapi.py"  # 2DP laptop
        spec = importlib.util.spec_from_file_location("lumapi", lumapi_path)
        lumapi_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lumapi_module)
        return lumapi_module
        
    def setup_simulation(self, mu, gamma, slg_overlap):
        if self.simulation_type == "FDTD":
            return self._setup_fdtd_simulation(mu, gamma)
        elif self.simulation_type == "FDE":
            return self._setup_fde_simulation(mu, gamma, slg_overlap)
        else:
            raise ValueError("Invalid simulation type. Supported types: 'FDTD', 'FDE'")
    
    def _setup_fdtd_simulation(self, mu, gamma):
        fdtd = self.lumapi_module.FDTD()
        
        # FDTD setup logic
        
        # Simulation region
        fdtd.addfdtd(dimension="3D", x=0.0e-9, y=0.0e-9, x_span=1.0e-6, y_span=1.0e-6, z=0.0e-9, z_span=0.5e-6)
        fdtd.set("simulation time", 500e-15)
        
        # Graphene surface conductivity model
        graphene_material = fdtd.addmaterial("Graphene")
        fdtd.setmaterial(graphene_material, "name", "Graphene")
        fdtd.setmaterial("Graphene", "scattering rate (eV)", gamma)
        fdtd.setmaterial("Graphene", "chemical potential (eV)", mu)
        
        fdtd.add2drect(name="Graphene", x=0.0e-9, y=0.0e-9, x_span=50.0e-6, y_span=50.0e-6, z=0.0e-9)
        fdtd.set("material", "Graphene")
        
        # Monitors for results
        fdtd.addpower(name="power", monitor_type=7, x=0.0e-9, y=0.0e-9, x_span=5.0e-6, y_span=5.0e-6, z=-1.0e-6)
        
        fdtd.save("mu_sweep_test_" + str(mu) + ".fsp")
        

        return fdtd.getresult("power", "T")
    
    def _setup_fde_simulation(self, mu, gamma, slg_overlap):
        mode = self.lumapi_module.MODE()
        
        
        # FDE setup logic
        
        # Simulation region
        mode.addfde(solver_type=2, x=0.0e-6, y=0.0e-6, x_span=2e-6, z=0.025e-6, z_span=1.2e-6)
        mode.set("x min bc", "Metal")
        mode.set("x max bc", "Metal")
        mode.set("z min bc", "Metal")
        mode.set("z max bc", "Metal")
        mode.set("min mesh step", 2e-9)
        mode.set("mesh cells x", 500)
        mode.set("mesh cells z", 600)
      
        # Waveguide
        mode.addrect(name="Silicon", x=0.0e-9, y=0.0e-9, z=0.0e-9, x_span=450e-9, y_span=1.0e-6, z_span=220e-9)
        mode.set("material", "Si (Silicon) - Palik")
      
        mode.addrect(name="SiO2", x=0.0e-9, y=0.0e-9, z=0.0e-9, x_span=3e-6, y_span=1.0e-6, z_min=-1e-6, z_max=120e-9)
        mode.set("material", "SiO2 (Glass) - Palik")
        mode.set("override mesh order from material database", 1)
        mode.set("mesh order", 3)
        
        # SLG1 surface conductivity model
        graphene_material = mode.addmaterial("Graphene")
        mode.setmaterial(graphene_material, "name", "SLG1")
        mode.setmaterial("SLG1", "scattering rate (eV)", gamma)
        mode.setmaterial("SLG1", "chemical potential (eV)", mu)
        # Add SLG 1
        mode.add2drect(name="SLG1", x=0.0e-6, y=0.0e-6, y_span=1e-6, z=120e-9)
        mode.set("material", "SLG1")
        mode.set("x span", slg_overlap)
        
        # Add gate dielectric
        mode.addrect(name="Al2O3", x=0.0e-6, y=0.0e-6, x_span=3e-6, y_span=1e-6, z_min=120e-9, z_max=160e-9)
        mode.set("material", "Al2O3 - Palik")
        
        # Add SLG 2
        mode.add2drect(name="SLG2", x=0.0e-6, y=0.0e-6, y_span=1e-6, z=160e-9)
        mode.set("material", "SLG1")
        mode.set("x span", slg_overlap)

        
        # Eigensolver parameters
        mode.setanalysis("wavelength", 1.55e-6)
        mode.setanalysis("search", "near n")
        mode.findmodes()
        
        # Return effective index and loss data for TE (mode1)
        n_eff = mode.getdata("mode1", "neff")
        real_n_eff = np.real(n_eff)[0]
        loss = mode.getdata("mode1", "loss") # dB/m
        
        # Append n_eff and loss values to a single text file
        result_filename = "FDE_results__gamma_mode1_" + str(gamma) + "_" + ".txt"
        write_header = not os.path.exists("FDE_results__gamma_mode1_" + str(gamma) + "_" + ".txt") or os.path.getsize("FDE_results__gamma_mode1_" + str(gamma) + "_" + ".txt") == 0
        with open(result_filename, "a") as file:
            if write_header:
                file.write("Chemical potential [eV],\tScattering rate [eV],\tEffective index,\tLoss [dB/m]\tSLG overlap [um]\n")
                file.write(f"{mu}\t{gamma}\t{real_n_eff[0]}\t{loss}\t{slg_overlap}\n")
            else:
                file.write(f"{mu}\t{gamma}\t{real_n_eff[0]}\t{loss}\t{slg_overlap}\n")
                
        # Return effective index and loss data for TM (mode2)
        n_eff = mode.getdata("mode2", "neff")
        real_n_eff = np.real(n_eff)[0]
        loss = mode.getdata("mode2", "loss") # dB/m
        
        # Append n_eff and loss values to a single text file
        result_filename = "FDE_results__gamma_mode2_" + str(gamma) + "_" + ".txt"
        write_header = not os.path.exists("FDE_results__gamma_mode2_" + str(gamma) + "_" + ".txt") or os.path.getsize("FDE_results__gamma_mode2_" + str(gamma) + "_" + ".txt") == 0
        with open(result_filename, "a") as file:
            if write_header:
                file.write("Chemical potential [eV],\tScattering rate [eV],\tEffective index,\tLoss [dB/m]\tSLG overlap [um]\n")
                file.write(f"{mu}\t{gamma}\t{real_n_eff[0]}\t{loss}\t{slg_overlap}\n")
            else:
                file.write(f"{mu}\t{gamma}\t{real_n_eff[0]}\t{loss}\t{slg_overlap}\n")

      
        # Save and run simulation
        mode.save("FDE_mu_gamma_SLGoverlap_" + str(mu) + "_" + str(gamma) + "_" + str(slg_overlap) + ".fsp")
        

        return real_n_eff, loss
    
    def run_simulations(self):
        for gamma in self.gamma_values:
            print(f"Scattering rate = {gamma}")
            for mu in self.mu_values:
                print(f"Fermi level = {mu}")
                for slg_overlap in self.slg_overlap:
                    print(f"SLG overlap = {slg_overlap}")
                    self.setup_simulation(mu, gamma, slg_overlap)
 
                
#===============================================================================================
    def _process_fde_results(self, gamma_values, hbar):
        
# Loss vs Ef figure ============================================================================
        plt.figure(dpi=600)
        for i,gamma in enumerate(gamma_values):
            result_filename = f"FDE_results__gamma_mode1_{gamma}_" + ".txt"
        
            if not os.path.exists(result_filename) or os.path.getsize(result_filename) == 0:
                print(f"No results found for gamma = {gamma}. Please run simulations first.")
                continue

            # Read data from the file
            data = np.loadtxt(result_filename, delimiter="\t", skiprows=1)

            # Extracting columns
            mu_values = data[:, 0]
            loss_values = data[:, 3]

            # Plot optical loss
            mu_loss_spline = make_interp_spline(mu_values, loss_values)
            mu_ = np.linspace(mu_values.min(), mu_values.max(), 30)
            loss_ = mu_loss_spline(mu_)
            # Use plt.cm.Blues colormap with varying alpha values
            color = plt.cm.Blues(i / (len(gamma_values) - 1)+0.3)  # Normalize i to range [0, 1]
            plt.plot(mu_, loss_ * 10**(-6),"o-", label=f"$\\tau$ = {hbar/gamma} s", color=color, alpha=0.7)
        
        plt.xlabel("E$_{F}$ [eV]")
        plt.ylabel("Optical loss [dB/$\mu$m]")
        plt.tick_params(direction="in", top=True, right=True)
        plt.legend()
        plt.show()

# Plot change in effective index ==============================================================

        plt.figure(dpi=600)
        for i,gamma in enumerate(gamma_values):
            result_filename = f"FDE_results__gamma_mode1_{gamma}_" + ".txt"
        
            if not os.path.exists(result_filename) or os.path.getsize(result_filename) == 0:
                print(f"No results found for gamma = {gamma}. Please run simulations first.")
                continue

            # Read data from the file
            data = np.loadtxt(result_filename, delimiter="\t", skiprows=1)

            # Extracting columns
            mu_values = data[:, 0]
            real_n_eff_values = data[:, 2]
            delta_n_eff = real_n_eff_values - real_n_eff_values[0]
            
            # Plot effective index
            mu_loss_spline = make_interp_spline(mu_values, delta_n_eff)
            mu_ = np.linspace(mu_values.min(), mu_values.max(), 30)
            delta_n_eff_ = mu_loss_spline(mu_)
            # Use plt.cm.Reds colormap with varying alpha values
            color = plt.cm.Reds(i / (len(gamma_values) - 1)+0.3)  # Normalize i to range [0, 1]
            plt.plot(mu_, delta_n_eff_, "o-", label=f"$\\tau$ = {hbar/gamma} s", color=color, alpha=0.7)
           
        
        plt.xlabel("E$_{F}$ [eV]")
        plt.ylabel("$\Delta$n$_{eff}$")
        plt.tick_params(direction="in", top=True, right=True)
        plt.legend()
        plt.show()
        
# Plot ER =============================================================================

        plt.figure(dpi=600)
        for i,gamma in enumerate(gamma_values):
            result_filename = f"FDE_results__gamma_mode1_{gamma}_" + ".txt"
        
            if not os.path.exists(result_filename) or os.path.getsize(result_filename) == 0:
                print(f"No results found for gamma = {gamma}. Please run simulations first.")
                continue

            # Read data from the file
            data = np.loadtxt(result_filename, delimiter="\t", skiprows=1)

            # Extracting columns
            mu_values = data[:, 0]
            loss_values = data[:, 3]
        
            # Calculating extinction ratio and phase shift as a function of length
            L = [10, 100, 200, 300, 400, 500]
            ER = []
            for j, l in enumerate(L):
                ER.append(l * (10**(-6)) * (max(loss_values) - min(loss_values)))
            
            # Plot ER and delta phi as a function of length
            # Use plt.cm.Greys colormap with varying alpha values
            color = plt.cm.Greys(i / (len(gamma_values) - 1)+0.3)  # Normalize i to range [0, 1]
            plt.plot(L,ER, "o-", label=f"$\\tau$ = {hbar/gamma} s", color=color, alpha=0.7)
            
        plt.xlabel("Length [$\mu$m]")
        plt.ylabel("ER [dB]")
        plt.tick_params(direction="in", top=True, right=True)
        plt.legend()
        plt.show()
           
# Plot phase shift ===================================================================
        plt.figure(dpi=600)
        for i,gamma in enumerate(gamma_values):
            result_filename = f"FDE_results__gamma_mode1_{gamma}_" + ".txt"
        
            if not os.path.exists(result_filename) or os.path.getsize(result_filename) == 0:
                print(f"No results found for gamma = {gamma}. Please run simulations first.")
                continue

            # Read data from the file
            data = np.loadtxt(result_filename, delimiter="\t", skiprows=1)

            # Extracting columns
            mu_values = data[:, 0]
            real_n_eff_values = data[:, 2]
            delta_n_eff = real_n_eff_values - real_n_eff_values[0]
            
            # Calculating extinction ratio and phase shift as a function of length
            L = [10, 100, 200, 300, 400, 500]
            delta_phi = []
            for j, l in enumerate(L):
                delta_phi.append(2 * delta_n_eff[12] * l / 1.55)
            
            # Plot ER and delta phi as a function of length
            # Use plt.cm.Greys colormap with varying alpha values
            color = plt.cm.Greys(i / (len(gamma_values) - 1)+0.3)  # Normalize i to range [0, 1]
            plt.plot(L,delta_phi,"o-", label=f"$\\tau$ = {hbar/gamma} s", color=color, alpha=0.7)
            
        plt.xlabel("Length [$\mu$m]")
        plt.ylabel("Phase shift [$\pi$]")
        plt.tick_params(direction="in", top=True, right=True)
        plt.legend()
        plt.show()
#==============================================================================================        

        # Append ER, delta n eff and delta phi values to a single text file
        modulators_data = zip(L, ER, delta_phi)
        result_filename = "modulator_results__gamma" + str(gamma) +".txt"
        write_header = not os.path.exists("modulator_results__gamma" + str(gamma) +".txt") or os.path.getsize("modulator_results__gamma" + str(gamma) +".txt") == 0
        with open(result_filename, "w") as file:
            write = csv.writer(file, delimiter="\t")
            if write_header:
                file.write("Length [um],\tExtinction ratio [dB],\tPhase shift [pi]\n")
                write.writerows(modulators_data)
            else:
                write.writerows(modulators_data)

        return L, ER, delta_phi

#=============================================================================================

class ElectronicCircuitAnalysis:
    def __init__(self):
        pass

    def analyze_circuit(self):
       vf = 0.95e+6  # m/s Fermi velocity
       hbar = 6.582e-16  # eV.s Planck constant
       hbar1 = 1.055e-34  # J.s Planck constant
       e0 = 1.602e-19  # C electronic charge
       eps0 = 8.854e-12  # F/m free space permitivty
       eps1 = 7.0  # relative permitivity of Al2O3+hBN
       dox = 20e-9  # dielectric thickness
       Lu = 0.4e-6  # Ungated graphene length
       wu = 20e-6  # Ungated graphene width
       Lg = 0.65e-6  # Gated graphene length
       wg = 20e-6  # Gated graphene width

       Cox = eps0 * eps1 / dox  # F/m2 oxide capacitance
       Cox1 = Cox * Lg * wg  # F total oxide capacitance

       Ef = 0.4  # eV Fermi energy in gated graphene
       Efu = 0.25  # eV Fermi energy in ungated graphene
       tao = 300e-15  # s Scattering time of gated graphene
       Rc = 600  # Ohm.um Contact resistance
       Rct = Rc / (wu * 1e+6)  # Ohm Total contact resistance

       Cq = 2 * e0 * e0 * Ef / (hbar * hbar1 * vf * vf * np.pi)  # F/m2 Quantum capacitance
       Cq1 = Cq * Lg * wg  # F total quantum capacitance
       Ceq = Cq1 * Cox1 / (2 * Cox1 + Cq1)  # F Equivalent total capacitance

       ff = np.arange(0.1e9, 200e9, 0.1e9)  # Hz Frequency

       Sig = (e0 * e0 / (np.pi * hbar * hbar1)) * (Ef * tao / (1 - 1j * 2 * np.pi * 0 * tao))  # S DC-conductivity of gated graphene
       mob = (e0 * vf * vf * tao / (Ef * 1.602e-19)) * 1e+4  # cm2/Vs Mobility as a function of Ef and tao
       Rgateds = 1 / Sig  # Ohm/sq Sheet resistance of gated graphene
       Rgated = Rgateds * (Lg / wg)  # Ohm Gated graphene total resistance

       taou = (mob * Efu * 1.602e-19) / (e0 * vf * vf * 1e+4)  # s Scattering time of ungated graphene
       Sigu = (e0 * e0 / (np.pi * hbar * hbar1)) * (Efu * taou / (1 - 1j * 2 * np.pi * 0 * taou))  # S DC-conductivity of ungated graphene
       Rungateds = 1 / Sigu  # Ohm/sq Ungated graphene sheet resistance
       Rungated = Rungateds * (Lu / wu)  # Ohm Ungated graphene total resistance

       Rt = 2 * (Rct + Rungated + Rgated)+50  # Total resistance of circuit and transmission line
       
       eo_bandwidth = (1/(2*np.pi*Rt*Ceq))*10**(-9)

       absVolrSquared = 1 / (1 + ((2 * np.pi) ** 2) * (ff ** 2) * (Rt ** 2) * (Ceq ** 2))  # Complex conjugate of voltage ratio Vm/Vd
       absVolr = np.sqrt(absVolrSquared)

       bb = (ff / ff) * (-3)

       plt.figure(dpi=600)
       plt.semilogx(ff, 10*np.log(absVolr),"k-", label=f"R$_c$ = {Rc} $\\Omega\\mu$m, C$_e$ = {Ceq} F")
       plt.semilogx(ff, bb, 'g--')
       plt.xlabel("Frequency [Hz]")
       plt.ylabel("EO bandwidth [dB]")
       plt.tick_params(direction="in", top=True, right=True)
       plt.tick_params(which="minor", direction="in")
       plt.legend()
       plt.show()
       
       data = np.column_stack((np.real(ff), np.real(10*np.log(absVolr))))
       np.savetxt("simulated_BW_SuSi/200umLength_80nmOx_eps7_1500nmGated_3500nmUngated_300fs_60Ohmum.txt", data, header='Frequency (Hz)\tEO Bandwidth (dB)', fmt='%.2e\t%.6f')
       
       return Ceq, Rt, eo_bandwidth, mob, taou, Rgateds, Rungateds, Cox

#=========================================================================================================================
# Run FDE simulations with different wavelength, mu, and gamma values
mu_values = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
#mu_values = [0., 0.05]
hbar = 6.5821e-16
tau_values = [350e-15]
slg_overlap_values = [0.55e-6]

gamma_values = [hbar / tau for tau in tau_values]

slg_overlap = [overlap for overlap in slg_overlap_values]

graphene_sim = GrapheneSimulation(simulation_type="FDE", mu_values=mu_values, gamma_values=gamma_values, slg_overlap=slg_overlap)
graphene_sim.run_simulations()

# Process FDE results for specific gamma values
gamma_to_process = gamma_values  # Choose a gamma value from your list
L, ER, delta_phi = graphene_sim._process_fde_results(gamma_to_process, hbar)

# Frequency response of equivalent circuit
#circuit_analysis = ElectronicCircuitAnalysis()
#Ceq, Rt, eo_bandwidth, mob, taou, Rgateds, Rungateds, Cox = circuit_analysis.analyze_circuit()
