import numpy as np
import matplotlib.pyplot as plt
# Όλα τα μεγέθη του κώδικα και οι εξισώσεις είναι στο SI
G = 6.67e-11            
c = 299792458.          
pi = np.pi
M_sun = 1.9891e30     
gamma0=gamma1=gamma2=gamma3=gamma4=gamma5=gamma6=0.0
rho_1=rho_2=rho_3=rho_4=rho_5=rho_6=0.0
K0=K1=K2=K3=K4=K5=K6=0.0
P0=P1=P2=P3=P4=P5=0.0
a1=a2=a3=a4=a5=a6=0.0
def set_eos(eos_name):
    global gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6
    global rho_1, rho_2, rho_3, rho_4, rho_5, rho_6
    global K0, K1, K2, K3, K4, K5, K6
    global P0, P1, P2, P3, P4, P5
    global a1, a2, a3, a4, a5, a6
    if eos_name == "GM1":
        gamma0, gamma1, gamma2 = 1.6356, 1.3082, 0.5103
        gamma3, gamma4, gamma5, gamma6 = 0.9431, 3.2095, 2.8973, 2.5144
        rho_1, rho_2, rho_3 = 10**(6.9626 + 3.0), 10**(11.4783 + 3.0), 10**(12.2341 + 3.0)
        rho_4, rho_5, rho_6 = 10**(13.6981 + 3.0), 10**(14.3853 + 3.0), 10**(14.9312 + 3.0)
        log_K0_cgs = 12.4928
        K0 = 10**(log_K0_cgs - 1.0 - 3.0 * gamma0)
    elif eos_name == "H3":
        gamma0, gamma1, gamma2 = 1.5950, 1.3021, 0.4741
        gamma3, gamma4, gamma5, gamma6 = 0.9455, 3.2473, 2.9180, 1.9421
        rho_1, rho_2, rho_3 = 10**(7.1558 + 3.0), 10**(11.5194 + 3.0), 10**(12.2298 + 3.0)
        rho_4, rho_5, rho_6 = 10**(13.7026 + 3.0), 10**(14.3214 + 3.0), 10**(14.6654 + 3.0)
        log_K0_cgs = 12.7365
        K0 = 10**(log_K0_cgs - 1.0 - 3.0 * gamma0)
    elif eos_name == "DH":
        gamma0, gamma1, gamma2 = 1.6021, 1.3030, 0.6165
        gamma3, gamma4, gamma5, gamma6 = 1.3397, 2.1052, 3.0053, 2.8605
        rho_1, rho_2, rho_3, rho_4, rho_5, rho_6 = 1.229e10, 3.649e14, 2.608e15, 1.012e17, 1.908e17, 9.124e17
        K0, K1, K2, K3, K4, K5, K6 = 7.842e6, 8.171e9, 8.114e19, 5.756e8, 5.528e-5, 1.543e-20, 6.154e-18
    if eos_name != "DH":
        K1 = K0 * rho_1**(gamma0 - gamma1)
        K2 = K1 * rho_2**(gamma1 - gamma2)
        K3 = K2 * rho_3**(gamma2 - gamma3)
        K4 = K3 * rho_4**(gamma3 - gamma4)
        K5 = K4 * rho_5**(gamma4 - gamma5)
        K6 = K5 * rho_6**(gamma5 - gamma6)
    P0 = K0*rho_1**gamma0
    P1 = K1*rho_2**gamma1
    P2 = K2*rho_3**gamma2
    P3 = K3*rho_4**gamma3
    P4 = K4*rho_5**gamma4
    P5 = K5*rho_6**gamma5
    a1 = P0*((1/(gamma0-1))-(1/(gamma1-1)))/(rho_1*c**2)
    a2 = ((a1*rho_2*c**2)+(P1*((1/(gamma1-1))-(1/(gamma2-1)))))/(rho_2*c**2)
    a3 = ((a2*rho_3*c**2)+(P2*((1/(gamma2-1))-(1/(gamma3-1)))))/(rho_3*c**2)
    a4 = ((a3*rho_4*c**2)+(P3*((1/(gamma3-1))-(1/(gamma4-1)))))/(rho_4*c**2)
    a5 = ((a4*rho_5*c**2)+(P4*((1/(gamma4-1))-(1/(gamma5-1)))))/(rho_5*c**2)
    a6 = ((a5*rho_6*c**2)+(P5*((1/(gamma5-1))-(1/(gamma6-1)))))/(rho_6*c**2)
def eos_baryon_from_pressure(P):
    if P <= 0.0: return 0.0, 0.0
    if P <= P0:
        rho = (P/K0)**(1/gamma0)
        epsilon = (rho * c**2) + (P/(gamma0-1))
        return float(epsilon), float(rho)
    elif P0 < P <= P1:
        rho = (P/K1)**(1/gamma1)
        epsilon = ((1+a1)*rho*c**2) + (P/(gamma1-1))
        return float(epsilon), float(rho)
    elif P1 < P <= P2:
        rho = (P/K2)**(1/gamma2)
        epsilon = ((1+a2)*rho*c**2) + (P/(gamma2-1))
        return float(epsilon), float(rho)
    elif P2 < P <= P3:
        rho = (P/K3)**(1/gamma3)
        epsilon = ((1+a3)*rho*c**2) + (P/(gamma3-1))
        return float(epsilon), float(rho)
    elif P3 < P <= P4:
        rho = (P/K4)**(1/gamma4)
        epsilon = ((1+a4)*rho*c**2) + (P/(gamma4-1))
        return float(epsilon), float(rho)
    elif P4 < P <= P5:
        rho = (P/K5)**(1/gamma5)
        epsilon = ((1+a5)*rho*c**2) + (P/(gamma5-1))
        return float(epsilon), float(rho)
    else:
        rho = (P/K6)**(1/gamma6)
        epsilon = ((1+a6)*rho*c**2) + (P/(gamma6-1))
        return float(epsilon), float(rho)
def get_dAlpha_dr(r, m, P_tot):
    if r <= 0: return 0.0
    num = G * (m + (4.0 * np.pi * r**3 * P_tot / c**2))
    den = r * (r*c**2 - 2.0*G*m)
    if den <= 0: return 0.0
    return float(num/den)
def tov_rhs(r, y):
    P_B, m_B, Phi = y
    eps_B, _ = eos_baryon_from_pressure(P_B)
    if P_B <= 0.0: P_B, eps_B = 0.0, 0.0
    dAlpha_dr = get_dAlpha_dr(r, m_B, P_B)
    dP_B = 0.0 if P_B == 0.0 else -(eps_B + P_B) * dAlpha_dr
    dm_B = 0.0 if P_B == 0.0 else 4.0 * pi * r**2 * eps_B / c**2
    return np.array([dP_B, dm_B, dAlpha_dr], dtype=np.float64)
def initialize(dr, P_Bc):
    eps_Bc, _ = eos_baryon_from_pressure(P_Bc)
    m_B0 = (4/3) * pi * dr**3 * (eps_Bc/c**2)
    return np.array([P_Bc, m_B0, 0.0], dtype=np.float64)
def integrate_tov(P_Bc, dr=5):
    r, y = dr, initialize(dr, P_Bc)
    history = []
    for _ in range(1, 500000):
        history.append(tuple(float(x) for x in (r, *y)))
        k1 = tov_rhs(r, y)
        k2 = tov_rhs(r + dr/2, y + dr*k1/2)
        k3 = tov_rhs(r + dr/2, y + dr*k2/2)
        k4 = tov_rhs(r + dr, y + dr*k3)
        y += dr*(k1 + 2*k2 + 2*k3 + k4)/6
        r += dr
        if y[0] <= 0.0:
            y[0] = 0.0
            return np.array(history)
    return np.array(history)
def calculate_mr_curve(P_Bc_array):
    R_vals, M_vals = [], []
    max_M_tot = 0.0 
    for P_Bc in P_Bc_array:
        sol = integrate_tov(P_Bc, dr=5) 
        if len(sol) < 20: continue
        R_star, M_star = sol[-1, 0] / 1000, sol[-1, 2] / M_sun
        if M_star > max_M_tot: max_M_tot = M_star
        elif max_M_tot > 0.5 and (max_M_tot - M_star > 0.015): break 
        R_vals.append(R_star)
        M_vals.append(M_star)
    return np.array(R_vals), np.array(M_vals)
if __name__ == "__main__":
    P_Bc_array = np.logspace(33.0, 36.0, 70) 
    plt.figure(figsize=(10, 8))
    eos_list = ["GM1", "H3",  "DH"]
    for eos_name in eos_list:
        print(f"\nΣχεδιασμός καμπύλης M-R για ({eos_name})...")
        set_eos(eos_name)
        R_baryonic, Mass_total = calculate_mr_curve(P_Bc_array)
        if len(R_baryonic) > 0:
            plt.plot(R_baryonic, Mass_total, label=f"{eos_name}", linewidth=2.0)
        else:
            print(f"  [!] Δεν βρέθηκαν σταθερά άστρα για την {eos_name}.")
    plt.xlabel("Radius $R$ (km)", fontsize=13)
    plt.ylabel("Total Mass $M$ ($M_\odot$)", fontsize=13)
    plt.title("Mass-Radius Relation (All EoS)", fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.xlim(9, 15)   
    plt.ylim(0.5, 2.6)
    plt.show()