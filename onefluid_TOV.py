import numpy as np
import matplotlib.pyplot as plt

# Όλα τα μεγέθη του κώδικα και οι εξισώσεις είναι στο SI
G = 6.67e-11         
c = 299792458.         
pi = np.pi
M_sun = 1.9891e30      

# Παράμετροι Καταστατικής Εξίσωσης Βαρυονικής Ύλης (Piecewise Polytrope)
gamma0, gamma1, gamma2 = 1.6021, 1.3030, 0.6165
gamma3, gamma4, gamma5, gamma6 = 1.3397, 2.1052, 3.0053, 2.8605
rho_1 = 1.229e10
rho_2 = 3.649e14
rho_3 = 2.608e15
rho_4 = 1.012e17
rho_5 = 1.908e17
rho_6 = 9.124e17

K0 = 7.842e6
K1 = 8.171e9
K2 = 8.114e19
K3 = 5.756e8
K4 = 5.528e-5
K5 = 1.543e-20
K6 = 6.154e-18

P0 = K0 * rho_1**gamma0
P1 = K1 * rho_2**gamma1
P2 = K2 * rho_3**gamma2
P3 = K3 * rho_4**gamma3
P4 = K4 * rho_5**gamma4
P5 = K5 * rho_6**gamma5

a1 = P0 * ((1/(gamma0-1))-(1/(gamma1-1))) / (rho_1*c**2)
a2 = ((a1*rho_2*c**2) + (P1*((1/(gamma1-1))-(1/(gamma2-1))))) / (rho_2*c**2)
a3 = ((a2*rho_3*c**2) + (P2*((1/(gamma2-1))-(1/(gamma3-1))))) / (rho_3*c**2)
a4 = ((a3*rho_4*c**2) + (P3*((1/(gamma3-1))-(1/(gamma4-1))))) / (rho_4*c**2)
a5 = ((a4*rho_5*c**2) + (P4*((1/(gamma4-1))-(1/(gamma5-1))))) / (rho_5*c**2)
a6 = ((a5*rho_6*c**2) + (P5*((1/(gamma5-1))-(1/(gamma6-1))))) / (rho_6*c**2)

def eos_baryon_from_pressure(P):
    """Υπολογίζει την πυκνότητα ενέργειας και τη μάζας ηρεμίας από την πίεση."""
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
    """Υπολογίζει το dPhi/dr = dAlpha/dr."""
    if r <= 0: return 0.0
    num = G * (m + (4.0 * pi * r**3 * P_tot / c**2))
    den = r * (r*c**2 - 2.0*G*m)
    if den <= 0: return 0.0
    return float(num/den)

def tov_rhs(r, y):
    """Οι εξισώσεις TOV (RHS) για ένα ρευστό."""
    P_B, m_B, Phi = y
    
    eps_B, _ = eos_baryon_from_pressure(P_B)
    
    if P_B <= 0.0:
        P_B = 0.0 
        eps_B = 0.0 
      
    dAlpha_dr = get_dAlpha_dr(r, m_B, P_B)
    
    dP_B = 0.0 if P_B == 0.0 else -(eps_B + P_B) * dAlpha_dr 
    dm_B = 0.0 if P_B == 0.0 else 4.0 * pi * r**2 * eps_B / c**2 
    dPhi = dAlpha_dr
    
    return np.array([dP_B, dm_B, dPhi], dtype=np.float64)

def initialize(dr, P_Bc):
    """Αρχικοποίηση στο κέντρο (r=dr)."""
    eps_Bc, _ = eos_baryon_from_pressure(P_Bc)
    m_B0 = (4/3) * pi * dr**3 * (eps_Bc/c**2)
    return np.array([P_Bc, m_B0, 0.0], dtype=np.float64)

def integrate_tov(P_Bc, dr=1):
    """Ολοκλήρωση με μέθοδο Runge-Kutta 4 (RK4)."""
    r = float(dr)
    y = initialize(dr, P_Bc)
    history = []
    
    for _ in range(1, 500000):
        history.append(tuple((r, *y)))
        
        # Αν η πίεση μηδενίστηκε, φτάσαμε στην επιφάνεια του αστέρα
        if y[0] <= 0.0:
            break
            
        k1 = tov_rhs(r, y)
        k2 = tov_rhs(r + dr/2, y + dr*k1/2)
        k3 = tov_rhs(r + dr/2, y + dr*k2/2)
        k4 = tov_rhs(r + dr, y + dr*k3)
        y += dr*(k1 + 2*k2 + 2*k3 + k4)/6
        r += dr
        
        if y[0] <= 0.0:
            y[0] = 0.0
            
    return np.array(history)

def find_Pc_for_target_mass(target_mass_sol, tol=1e-4):
    """Βρίσκει την κεντρική πίεση που δίνει την επιθυμητή μάζα με bisection."""
    P_vals = np.logspace(33.0, 35.5, 50)
    P_min, P_max = None, None
    prev_mass = 0.0
    
    # 1. Βρίσκουμε το διάστημα (bracket)
    for i in range(len(P_vals)):
        sol = integrate_tov(P_vals[i], dr=10) # Μεγάλο dr για γρήγορο ψάξιμο
        curr_mass = sol[-1, 2] / M_sun
        if curr_mass < prev_mass: # Περάσαμε το M_max (turn-over point)
            break
        if prev_mass <= target_mass_sol and curr_mass >= target_mass_sol:
            P_min = P_vals[i-1]
            P_max = P_vals[i]
            break
        prev_mass = curr_mass
        
    if P_min is None:
        return None
        
    # 2. Bisection για ακρίβεια
    for _ in range(50):
        P_mid = 0.5 * (P_min + P_max)
        sol = integrate_tov(P_mid, dr=1)
        current_mass = sol[-1, 2] / M_sun
        if abs(current_mass - target_mass_sol) < tol:
            return P_mid
        if current_mass > target_mass_sol:
            P_max = P_mid
        else:
            P_min = P_mid
            
    return P_mid

# ==========================================
# ΕΚΤΕΛΕΣΗ ΚΑΙ ΣΧΕΔΙΑΣΜΟΣ (MAIN)
# ==========================================
target_mass = 1.44

print(f"Υπολογισμός για Αμιγώς Βαρυονικό Αστέρα {target_mass} M_sun...")
P_Bc_ideal = find_Pc_for_target_mass(target_mass)

if P_Bc_ideal is not None:
    # Ολοκλήρωση με μικρό βήμα για ομαλό διάγραμμα
    sol = integrate_tov(P_Bc_ideal, dr=10)
    r_vals = sol[:, 0] / 1000  # Μετατροπή σε km
    P_B_vals = sol[:, 1]
    
    R_B = r_vals[-1]
    
    # Εξαγωγή του προφίλ πυκνότητας (rho)
    rho_B_vals = np.zeros_like(r_vals)
    for i in range(len(r_vals)):
        eps_B, rho_B = eos_baryon_from_pressure(P_B_vals[i])
        rho_B_vals[i] = rho_B
        
    print(f"-> Επιτυχία! R_B = {R_B:.2f} km, M = {sol[-1, 2]/M_sun:.4f} M_sun")
    
    # Σχεδιασμός του Διαγράμματος
    plt.figure(figsize=(10, 6))
    plt.plot(r_vals, rho_B_vals, label="Pure Baryonic", color='black', linewidth=2)
    
    plt.xlabel("Radius (km)")
    plt.ylabel("Density [kg/m³]") 
    plt.title(f"Density Profile for Pure Baryonic {target_mass} $M_\odot$ Star")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.show()
else:
    print("Αποτυχία εύρεσης αστέρα με αυτή τη μάζα.")