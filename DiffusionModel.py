import scipy.special as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from lmfit import minimize, Parameters
from findiff import FinDiff
import json

class Diffusion:
    #contains different options to calculate the concentration profile.
    def __init__(self):
        self.concentration = None
        self.concentration_profile = None
        self.diffusivity = None

    # 1D erf Solution with the derivative solved either analytically or discrete
    def oned_discrete_solution(self, params, t, x, temp):
        d0 = params['max_diffusivity']
        a = 0.3832610265152188#params['diffusivity_prefactor']#
        Ea = 67299#params['activation_energy']#
        c0 = params['concentration']


        R = 8.314  # J/mol*T
        D = a * d0 * np.exp(- Ea / (R * temp))
        dx = x[1] - x[0]
        arg = np.outer(1 / np.sqrt(D * t), x)
        c = -c0 * sp.erf(arg)
        self.concentration_profile = c * dx
        self.concentration = 2 * np.sum(self.concentration_profile, axis=1)
        self.diffusivity = D

        return self.concentration

    def oned_analytical_solution(self, params, t, temp):
        d0 = params['max_diffusivity']
        a = params['diffusivity_prefactor']
        Ea = params['activation_energy']
        x1 = params['upper_bound']
        x0 = params['lower_bound']
        c0 = params['concentration']


        R = 8.314  # J/mol*T
        D = a * d0 * np.exp(- Ea / (R * temp))

        upper_arg1 = x1 * sp.erf(x1 / np.sqrt(D * t))
        upper_arg2 = np.sqrt(D * t / np.pi) * np.exp(- x1 * x1 / (D * t))
        lower_arg1 = x0 * sp.erf(x0 / np.sqrt(D * t))
        lower_arg2 = np.sqrt(D * t / np.pi) * np.exp(- x0 * x0 / (D * t))

        self.concentration = 2 * c0 * (upper_arg1 + upper_arg2 - lower_arg1 - lower_arg2)

        return self.concentration

    # 3D ridial solution for a sphere
    def radial_solution(self, params, t, r, temp):
        r0 = params['diameter']/2
        d0 = params['max_diffusivity']
        a = params['diffusivity_prefactor']
        Ea = params['activation_energy']
        c0 = params['concentration']
        R = 8.314  # J/mol*T
        dr = r[1] - r[0]

        D = a * d0 * np.exp(- Ea / (R * temp))
        T = D * t

        arg1 = np.outer((r0 + r), 1 / (2 * np.sqrt(T)))
        summand1 = c0 / 2 * (sp.erf(arg1) + sp.erf(arg1))
        prefac = np.outer(c0 / r, np.sqrt(T / np.pi))
        arg2_plus = np.outer((r0 + r)**2, 4*T)
        arg2_minus = np.outer((r0 - r)**2, 4*T)
        summand2 = prefac * (np.exp(- arg2_minus) - np.exp(- arg2_plus))
        c = summand1 - summand2

        self.concentration_profile = c * dr
        self.concentration = 2*np.sum(self.concentration_profile, axis=0)

        return self.concentration

    def threed_discrete_solution(self, params, t, x, temp):
        d0 = params['max_diffusivity']
        a = params['diffusivity_prefactor']
        Ea = params['activation_energy']
        correction_factor = 2.2842069766076842
        c0 = 0.14444444444e9 * correction_factor
        sigma = params['sigma']
        R = 8.314  # J/mol*T
        D = a * d0 * np.exp(- Ea / (R * temp))
        dx = x[1] - x[0]
        j = sp.jv(0, x*np.sqrt(-sigma/D))
        t_arg = np.exp(sigma * t)
        c = c0 * np.outer(j, t_arg)
        self.concentration_profile = c*dx
        self.concentration = np.sum(self.concentration_profile, axis=0)

        return self.concentration

    # 1D Plane Sheet Approximatiin
    def oned_sum_solution(self, params, t, x, temp):
        d0 = params['max_diffusivity']
        a =  0.010363806282461566#params['diffusivity_prefactor']
        Ea = 59525.36041776226#params['activation_energy']
        c0 = params['concentration']
        d = params['diameter']
        R = 8.314  # J/mol*T
        D = a * d0 * np.exp(- Ea / (R * temp))
        dx = x[1] - x [0]
        prefac = 4 * c0 / np.pi

        summand = 0
        sum = 0
        b_prev = 1
        c_prev = 1
        for ii in range(500):
            a = 1 / (2 * ii + 1)
            b = np.sin((2 * ii + 1) * np.pi * x / d)
            c = np.exp(- (2 * ii + 1)**2 * np.pi**2 * D * t / (d**2))
            summand = a * np.outer(b, c)
            sum += summand

            b_prev = b
            c_prev = c

        self.concentration_profile = prefac * sum * dx
        self.concentration = np.sum(self.concentration_profile, axis=0)
        return self.concentration

    def fitting(self, params, t, data, x, temp, l0, ref_data):

        #------------------------------------------------------------
        # Chose the concentration profile from the solutions above
        #------------------------------------------------------------

        #c_tot = self.oned_discrete_solution(params, t, x, temp)
        c_tot = self.threed_discrete_solution(params, t, x, temp)
        #c_tot = self.oned_sum_solution(params, t, x, temp)
        #c_tot = self.radial_solution(params, t, x, temp)

        # ------------------------------------------------------------
        # reference data needed for 1D Solution fitting
        # ------------------------------------------------------------
        fitting_data = data - l0 #- ref_data

        pl_fit_coeff = params['plasmon_fitting_coefficient']
        pl_fit_offset = params['plasmon_fitting_offset']
        offset = params['offset']

        model = pl_fit_coeff * c_tot + pl_fit_offset
        model = model - model[0] + offset
        return np.abs(fitting_data-model)

    def diff_fitting(self, params, t, data, x, temp, ref):
        c_tot = self.oned_discrete_solution(params, t, x, temp)
        pl_fit_coeff = params['plasmon_fitting_coefficient']
        c_tot_diff = differentiate_array([t, c_tot])
        data_diff = differentiate_array([t, data]) - differentiate_array([t, ref])
        model = pl_fit_coeff * c_tot_diff
        return np.abs(data_diff - model)

class VacuumData:
    # reference data from the in-vacuum heating experiment
    def __init__(self, peak_length):
        self.data = None
        self.size = None
        self.peak_length = peak_length
        self.load_data()

    def load_data(self):
        df = pd.read_csv(
            '/Users/Joeme/Library/Mobile Documents/com~apple~CloudDocs/ETH/MSc/Masterarbeit/Plots/SpectumMeasurements/20200226_Ga-doped_Au_JSAu21_high_T_evolution.csv')
        l_temp = df["wavelength"].to_numpy()[:101]
        diff = self.relative_growth(l_temp[-1], df["wavelength"][101:self.peak_length])
        l_temp = l_temp - l_temp[0]
        factor = np.exp(-df["time"].to_numpy() / 80)[101:self.peak_length]
        summand = (diff * factor + l_temp[-1])[:]
        self.data = np.append(l_temp, summand)
        self.size = len(l_temp)

    def relative_growth(self, l0, data):
        return data.to_numpy() - l0

#-------------------------------------------------------------
# Fit the material parameters of the diffusion equation such
# that the np.abs(fitting_data-model) is minimized

# chose either fitting or diff_fitting (fits the derivative
# of the data)
#------------------------------------------------------------
def diffusion_fitting():
    with open('params.json') as data:
        params = json.load(data)

    df = pd.read_csv(params['file'])
    t = (df["time"].to_numpy()[101:] * 60)
    t = t - t[0]
    temp = df["temperature"].to_numpy()[101:] + 273.15
    peak = df["wavelength"].to_numpy()[:]
    l0 = peak[0]
    peak = peak[101:]
    jj = -1000
    ii = -1
    #x = np.linspace(-4.5, 0, 1000) * 1e-9  # m  for 1D discrete
    #x = np.linspace(0, 9, 1000) * 1e-9    # for 1D plane sheet (summed)
    x = np.linspace(0, 4.5, 1000) * 1e-9   # for 3D radial
    diffusion = Diffusion()
    VacumHeating = VacuumData(len(peak))
    params = Parameters()
    params.add('plasmon_fitting_coefficient', value=-7.93, min=-10.8, max=-7.93, vary=False) #fit bis 62
    params.add('plasmon_fitting_offset', value=556.651865, vary=False)
    params.add('max_diffusivity', value=4.5e-12, vary=False)
    params.add('diffusivity_prefactor', value=1, min=1e-1, max=1, vary=False)
    params.add('activation_energy', value=67299, min=44000, max=2e5, vary=False)
    params.add('concentration', value=5.15e9/9, vary=False)
    params.add('sigma', value=-0.00008, min=-0.6e-4, max=-1.2e-4)
    params.add('diameter', value=9e-9, vary=False)
    params.add('tempconst', value=0.0134, vary=False)
    params.add('offset', value=33.6, min=30, max=40)

    for method in ['powell', 'nelder']:
        out = minimize(diffusion.fitting, params, args=(t[jj:ii], peak[jj:ii], x, np.mean(temp[0:1000]), l0, VacumHeating.data[jj:ii]), method=method)

        # differential fitting
        #out = minimize(diffusion.diff_fitting, params, args=(t[jj:ii], peak[jj:ii], x, temp[jj:ii], VacumHeating.data[jj:ii]), method=method)
        a = out.params['diffusivity_prefactor'].value
        b = out.params['activation_energy'].value
        c = out.params['sigma'].value
        d = out.params['plasmon_fitting_coefficient'].value
        e = out.params['offset'].value
        print("Method:", method, "Diffusivity Prefactor =", a, "Activation Energy =", b, "Sigma = ", c,
              "Offset = ", e)
        fitting_result_analysis(x, a, b, c, d, e, ii, jj, df, l0, VacumHeating.data[jj:ii])

#-------------------------------------------------------------
# prints the Average and Standard Deviation of the fitted
# np.abs(fitting_data-model)
#------------------------------------------------------------
def fitting_result_analysis(x, a, b, c, d, e, ii, jj, df, l0, ref):
    t = df["time"].to_numpy()[101:] * 60
    temp = df["temperature"].to_numpy()[101:] + 273.15
    peak = df["wavelength"].to_numpy()[101:]
    with open('params.json') as data:
        params = json.load(data)
    params['diffusivity_prefactor'] = a
    params['activation_energy'] = b  # 67299                  # J/mol Ga in Au
    params['sigma'] = c
    params["plasmon_fitting_coefficient"] = d
    params['offset'] = e
    ga_in_au = Diffusion()
    diff = ga_in_au.fitting(params, t[jj:ii], peak[jj:ii], x, np.mean(temp[100:200]), l0, ref)
    #diff = ga_in_au.diff_fitting(params, t[jj:ii], peak[jj:ii], x, np.mean(temp[jj:ii]), ref)
    print(np.mean(diff), np.std(diff), np.max(diff))

#------------------------------------------------------------
# returns LSPR wavelength for given doping concentration
#------------------------------------------------------------
def concentration_to_wavelength(params, c_tot):
    a = params["plasmon_fitting_coefficient"]
    l0 = params["plasmon_fitting_offset"]
    return a * c_tot + l0

#------------------------------------------------------------
# plot simulation data vs. measured data
# chose measured data via params file
#------------------------------------------------------------
def oned_diffusion_plot():
    with open('params.json') as data:
        params = json.load(data)

    df1 = pd.read_csv(params['file'])
    t = df1["time"].to_numpy()[1:]*60
    temp = df1["temperature"].to_numpy()[1:]+273.15
    peak = df1["wavelength"].to_numpy()[1:]

    GaInAu = Diffusion()
    VacuumHeating = VacuumData(len(peak))
    l0 = peak[0]

    # ------------------------------------------------------------
    # 1D discrete solution
    # ------------------------------------------------------------
    #x = np.linspace(-4.5, 0, 1000)*1e-9
    #GaInAu.oned_discrete_solution(params, t, x, temp)

    # ------------------------------------------------------------
    # 1D plane sheet approximation solution
    # ------------------------------------------------------------
    x = np.linspace(0, 9, 1000)*1e-9
    GaInAu.oned_sum_solution(params, t, x, temp)

    #data range
    ii = 10
    jj = 30
    # ------------------------------------------------------------
    # plot wavelength evolution over time for given diffusion model
    # ------------------------------------------------------------
    wl_sim = concentration_to_wavelength(params, GaInAu.concentration)
    diffusion_plot(params, t[ii:jj], (peak-l0)[ii:jj], (wl_sim-wl_sim[0] + VacuumHeating.data)[ii:jj])
    diffusion_plot(params, t, (peak-l0), (wl_sim-wl_sim[0] + VacuumHeating.data))

    # ------------------------------------------------------------
    # plot wavelength evolution over time for given diffusion model
    # but for derivative mode
    # ------------------------------------------------------------
    #wl_diff = differentiate_array([t, wl_sim]) + differentiate_array([t, VacuumHeating.data])
    #peak_diff = differentiate_array([t, peak])
    #diffusion_plot(params, t[ii:jj], peak_diff[ii:jj], wl_diff[ii:jj])
    #diffusion_plot(params, t[1:], peak_diff[:], wl_diff[:])

    # ------------------------------------------------------------
    # plot concentration profile evolution over time for given diffusion model
    # ------------------------------------------------------------
    #concentration_profile_plot(GaInAu.concentration_profile.transpose())

#------------------------------------------------------------
# plots time vs. LSPR wavelength data and saves as .svg file
# set title in params file
#------------------------------------------------------------
def diffusion_plot(params, t, peak, wl_sim, save=False):
    fig, ax = plt.subplots()
    ax.scatter(t, peak, color='#A8322D', s=1, alpha=1,
               label='$\lambda_{LSP, JSAu26}$')
    ax.scatter(t, wl_sim, color='#1F407A', s=1, alpha=1, label=params["label"])

    ax.set_xscale('log')
    ax.set_xlim([10, 5e5])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wavelength (nm)')
    ax.legend()
    plt.title('Ga-doped Au Oxidation Plasmon Peak Evolution and Simulation')
    plt.savefig(fname=params["title"] + ".svg", type='svg')
    plt.show()

#------------------------------------------------------------
# discrete array differentiation
# could also use FinDiff method
#------------------------------------------------------------
def differentiate_array(data):
    t = data[0]
    wl = data[1]
    #d_dt = FinDiff(0, t[1]-t[0])
    #return d_dt(wl)
    dt = t[1]-t[0]
    wl_diff = np.zeros(len(wl)-1)
    for ii in range(len(wl)-1):
        dwl = wl[ii+1] - wl[ii]
        wl_diff[ii] = dwl / dt
    return wl_diff

#------------------------------------------------------------
# plots all the simulation results for set parameter vs.
# measurements
#------------------------------------------------------------
def combined_model_plot():
    with open('params.json') as data:
        params = json.load(data)

    df1 = pd.read_csv(params['file'])
    t = df1["time"].to_numpy()[1:]*60
    T = df1["temperature"].to_numpy()[1:]
    temp = T +273.15
    peak = df1["wavelength"].to_numpy()[1:]
    peak = peak - peak[0]
    offset = params['offset']
    VacuumHeating = VacuumData(len(peak))
    x = np.linspace(-4.5, 0, 1000)*1e-9
    x2 = np.linspace(0, 9, 1000) * 1e-9
    GaInAu1D = Diffusion()
    GaInAu3D = Diffusion()

    GaInAu3D.threed_discrete_solution(params, t, x, np.mean(temp[200:1000]))
    wl_sim3D = concentration_to_wavelength(params, GaInAu3D.concentration)
    wl_sim3D = wl_sim3D - wl_sim3D[0] + offset
    GaInAu1D.oned_discrete_solution(params, t, x, temp)
    wl_sim1D = concentration_to_wavelength(params, GaInAu1D.concentration)
    wl_sim1D = wl_sim1D - wl_sim1D[0] + VacuumHeating.data
    GaInAu1D.oned_sum_solution(params, t, x2, temp)
    wl_sim1Dtot = concentration_to_wavelength(params, GaInAu1D.concentration)
    wl_sim1Dtot = wl_sim1Dtot - wl_sim1Dtot[0] + VacuumHeating.data

    fig, ax_temp = plt.subplots()
    ax = ax_temp.twinx()

    #oned_sum_string = '1D: $\\frac{4c_0}{\pi}\sum_{j=0}^{\u221e}{\\frac{1}{2j+1}\sin{\\frac{(2j+1)\pi x}{L}}' \
    #                  '\exp{-\\frac{(2j+1)^2\pi^2Dt}{L^2}}}$'
    oned_sum_string = 'Plane-Sheet Approx.'

    ax_temp.scatter(t, T, color='#A8322D', s=1, alpha=0.5, label='Temperature')
    ax.scatter(t, peak, color='black', s=1.5, alpha=1,
               label='$\lambda_{peak, JSAu23}$')
    ax.scatter(t, wl_sim1D, color='#1F407A', s=1, alpha=1, label='1D Early Stage Approx') #label="1D: -erf(x/$\sqrt{Dt}$)")
    ax.scatter(t[150:], wl_sim3D[150:], color='#0069B4', s=1, alpha=0.8, label='3D Late Stage Approx')
    ax.scatter(t[0:150], wl_sim3D[0:150], color='#0069B4', s=1, alpha=0.3)# label='3D Late Stage Approx')#label="3D: $J_0(\sqrt{\\frac{\sigma}{D}}x)*\exp{-\sigma t}$")
    ax.scatter(t, wl_sim1Dtot, color='#3C5A0F', s=1, alpha=1, label=oned_sum_string)

    ax_temp.yaxis.tick_right()
    ax.yaxis.tick_left()
    ax_temp.yaxis.set_label_position("right")
    ax.yaxis.set_label_position("left")
    ax.set_xlim([10, 6e4])
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Peak Shift $\Delta\lambda_{peak}$ (nm)')
    ax_temp.set_ylabel('Temperature (°C)')
    ax_temp.set_ylim(top=240)
    ax.legend(loc=0)
    ax_temp.legend(loc=4)
    plt.title('Ga-doped Au Oxidation Plasmon Peak Evolution and Simulation')
    plt.savefig(fname=params["title"] + ".svg", type='svg')
    plt.show()

#------------------------------------------------------------
# plots the concentration profile according to the selected
# diffusion method. input data index and name it with the
# respective time stamp from the measurement file.
#------------------------------------------------------------
def concentration_profile_plot(profile):
    fig, ax = plt.subplots()
    #x = np.linspace(0, 4.5, 1000)     # for 1D erf solution and 3D radial solution
    x = np.linspace(0, 9, 1000)        # for 1D plane sheet approx.
    dc = 4.05 / 4.05395750e-03

    # ------------------------------------------------------------
    # create two arrays. one with index ii corresponding to a
    # spectrum measurement at time t (second array)
    # ------------------------------------------------------------
    #[0, 13, 23, 100, 994, 1372] ["0sec", "10.5min", "20.8min", "100min", "17h", "23.5h"]
    #[75, 121, 239, 1372] ["37min", "1h", "2h", "23.5h"]
    for ii in zip([5, 16, 24, 40, 138, 624], ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5))], ["1.5min", "7.1min", "11.2min", "19.3min", "69min", "315.3min"]):
            ax.plot(x, profile[ii[0]]*dc, color='#1F407A', linestyle=ii[1], label=ii[2])
    ax.add_patch(patches.Rectangle(xy=(0, 0), width=9, height=6, alpha=0.2, color='grey', fill=True))
    ax.add_patch(patches.Rectangle(xy=(-1, 0), width=1, height=6, color='grey', fill=True))
    ax.add_patch(patches.Rectangle(xy=(9, 0), width=1, height=6, color='grey', fill=True))
    ax.text(0.92, 0.15, "Shell", transform=ax.transAxes, fontsize=11)
    ax.text(0.01, 0.15, "Shell", transform=ax.transAxes, fontsize=11)
    ax.text(0.45, 0.15, "Core", transform=ax.transAxes, fontsize=11)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 4.5])
    ax.set_xticks(np.linspace(-1, 10, 12))
    ax.legend(loc=1)
    ax.set_xlabel('Radius (nm)')
    ax.set_ylabel('Relative Dopant Concentration (%)')
    plt.title('Doping Concentration Profile 1D - Plane-Sheet Approximation')
    plt.savefig(fname='Doping Concentration Profile 1D - Plane-Sheet Approximation' + ".svg", type='svg')
    plt.show()
