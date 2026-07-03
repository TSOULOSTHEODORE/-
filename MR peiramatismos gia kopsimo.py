import numpy as np
import matplotlib.pyplot as plt

# Όλα τα μεγέθη του κώδικα και οι εξισώσεις είναι στο SI
G = 6.67e-11           
c = 299792458.          
pi = np.pi
M_sun = 1.9891e30      


m_x   = 1.42e-27
gx_mf = 5.61e29 
hbar  = 1.05457e-34
c_pre_x = (0.5*(gx_mf)**2)*(hbar**3)/c

gamma0, gamma1, gamma2 = 1.6021, 1.3030, 0.6165
gamma3, gamma4, gamma5, gamma6 = 1.3397, 2.1052, 3.0053, 2.8605

rho_1 = 1.229e10
rho_2 = 3.649e14
rho_3 = 2.608e15
rho_4 = 1.012e17
rho_5 = 1.908e17
rho_6=  9.124e17



K0 = 7.842e6
K1 = 8.171e9
K2 = 8.114e19
K3 = 5.756e8
K4 = 5.528e-5
K5 = 1.543e-20
K6 = 6.154e-18

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
def eos_dm_from_pressure(P):
    if P is None or P <= 0.0:
        return 0.0, 0.0
    n = np.sqrt(P / c_pre_x) 
    epsilon = m_x * c**2 * n + c_pre_x * n**2
    return float(epsilon), float(n)

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
    P_B, P_DM, m_B, m_DM, Phi = y
    eps_B, _ = eos_baryon_from_pressure(P_B)
    eps_DM, _ = eos_dm_from_pressure(P_DM)
    if P_B <= 0.0:
        P_B = 0.0
        eps_B = 0.0

    if P_DM <= 0.0:
        P_DM = 0.0
        eps_DM = 0.0
        dP_DM = 0.0
        dm_DM = 0.0
    P_tot = P_B + P_DM
    m_tot = m_B + m_DM
    dAlpha_dr = get_dAlpha_dr(r, m_tot, P_tot)
    dP_B  = 0.0 if P_B == 0.0 else -(eps_B + P_B) * dAlpha_dr
    dP_DM = 0.0 if P_DM == 0.0 else -(eps_DM + P_DM) * dAlpha_dr

    dm_B  = 0.0 if P_B == 0.0 else 4.0 * pi * r**2 * eps_B / c**2
    dm_DM = 0.0 if P_DM == 0.0 else 4.0 * pi * r**2 * eps_DM / c**2
    dPhi = dAlpha_dr
   
    return np.array([dP_B, dP_DM, dm_B, dm_DM, dPhi], dtype=np.float64)


def initialize(dr, P_Bc, P_DMc):
    eps_Bc,_ = eos_baryon_from_pressure(P_Bc)
    eps_DMc,_ = eos_dm_from_pressure(P_DMc)
    m_B0 = (4/3) * pi * dr**3 * (eps_Bc/c**2)
    m_DM0 = (4/3) * pi * dr**3 * (eps_DMc/c**2)
    return np.array([P_Bc, P_DMc, m_B0, m_DM0, 0.0], dtype=np.float64)


def integrate_tov(P_Bc, P_DMc, dr=25):
    r = dr
    y = initialize(dr, P_Bc, P_DMc)
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


        if y[1] <= 0.0:
            y[1] = 0.0
        

        if y[0] == 0.0 and y[1] == 0.0:
            return np.array(history)
    return np.array(history)


def apply_metric_shift(history):
    r, P_B, P_DM, m_B, m_DM, Phi = history[-1]
    m_tot = m_B + m_DM
    A_surface = 0.5*np.log(1 - 2*G*m_tot/(r*c**2)) - Phi
    history[:,5] += A_surface
    return history



def mass_ratio(P_Bc, P_DMc):
    
    sol = integrate_tov(P_Bc, P_DMc)  

    if len(sol) == 0:
        return np.inf  

    m_B = sol[-1, 3]
    m_DM = sol[-1, 4]

    if m_B <= 0:
        return np.inf  

    return m_DM / (m_B+ m_DM)


def find_bracket(P_Bc, target_ratio):
    P_vals = np.logspace(25, 35,40)  
    prev_diff = mass_ratio(P_Bc, P_vals[0]) - target_ratio

    for i in range(1, len(P_vals)):
        ratio = mass_ratio(P_Bc, P_vals[i])
        diff = ratio - target_ratio

        
        if prev_diff * diff < 0.0 :
            return P_vals[i-1], P_vals[i]  

        prev_diff = diff

    return None, None

def find_P_DMc(P_Bc, target_ratio, tol=1e-8, max_iter=40):
    P_min, P_max = find_bracket(P_Bc, target_ratio)
    
    if P_min is None:
        print("No solution for P_Bc =", P_Bc)
        return None

    for i in range(max_iter):
        P_mid = 0.5*(P_min + P_max)
        ratio = mass_ratio(P_Bc, P_mid)

        if ratio > target_ratio:
            P_max = P_mid
        else:
            P_min = P_mid

       
        if abs(ratio - target_ratio) < tol:
            break

    return P_mid
def calculate_mr_curve(P_Bc_array, target_fraction):
    R_B_vals = []
    M_tot_vals = []
    
    total_stars = len(P_Bc_array)
    
    max_M_tot = 0.0 
    
    for idx, P_Bc in enumerate(P_Bc_array):
        if target_fraction == 0.0:
            P_DMc = 0.0  
        else:
            P_DMc = find_P_DMc(P_Bc, target_fraction)
          
            if P_DMc is None: 
                continue 
                
       
        sol = integrate_tov(P_Bc, P_DMc, dr=5) 
        
       
        if len(sol) < 20: 
            continue
            
        
        idx_B = np.where(sol[:, 1] <= 0.0)[0]
        if len(idx_B) > 0:
            R_B = sol[idx_B[0], 0] / 1000 
        else:
            R_B = sol[-1, 0] / 1000
       
        M_tot = (sol[-1, 3] + sol[-1, 4]) / M_sun
        
       
        if M_tot > max_M_tot:
            
            max_M_tot = M_tot
        elif max_M_tot - M_tot > 0.005:
            
            print(f"    [!] Μέγιστη Μάζα επιτεύχθηκε ({max_M_tot:.3f} M_sun). Διακοπή καμπύλης!")
            break 
   
        R_B_vals.append(R_B)
        M_tot_vals.append(M_tot)
        
        
        if (idx + 1) % 10 == 0:
            print(f"    Πρόοδος: {idx+1}/{total_stars} άστρα υπολογίστηκαν.")
            
    return np.array(R_B_vals), np.array(M_tot_vals)


P_Bc_array = np.logspace(33.5) 

fractions = [0.0, 0.00035, 0.00069, 0.0010, 0.0014, 0.0035, 0.01, 0.05, 0.1]

plt.figure(figsize=(10, 8))

for F_chi in fractions:
    print(f"\nΣχεδιασμός καμπύλης M-R για {F_chi*100:.3f}% Σκοτεινή Ύλη...")
    
    R_baryonic, Mass_total = calculate_mr_curve(P_Bc_array, F_chi)
    
    if len(R_baryonic) > 0:
        if F_chi == 0.0:
            plt.plot(R_baryonic, Mass_total, label="Pure Baryonic (0%)", color='black', linestyle='--', linewidth=2.5)
        else:
            plt.plot(R_baryonic, Mass_total, label=f"DM {F_chi*100:.3f}%")
    else:
        print(f"  [!] Δεν βρέθηκαν σταθερά άστρα για {F_chi*100:.3f}%")

plt.xlabel("Baryonic Radius $R_B$ (km)", fontsize=13)
plt.ylabel("Total Mass $M_{tot}$ ($M_\odot$)", fontsize=13)
plt.title(f"Mass-Radius Relation", fontsize=15)
plt.grid(True)
plt.legend()
plt.xlim(9, 15)   
plt.ylim(0.5, 2.6)
plt.show()