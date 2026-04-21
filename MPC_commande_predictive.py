# =============================================================================
#  MPC_commande_predictive.py
#  Commande Prédictive Simple (MPC) — SANS TOOLBOX — Python / CVXPY
#
#  Système : Modèle mécanique PMSM  →  G(s) = 1 / (J·s + f)
#  Méthode : Model Predictive Control (MPC) via optimisation convexe (CVXPY)
#
#  Auteur  : RAKOTONAMBININA Jehovamiahy Tsiorintsoa
#  Filière : M2 Mécatronique — ESP Antsiranana
#  Cours   : Asservissement Numérique
#  Date    : Avril 2026
# =============================================================================
#
#  Installation des dépendances :
#      pip install cvxpy numpy matplotlib scipy
#
#  Exécution :
#      python MPC_commande_predictive.py
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cvxpy as cp
from scipy.linalg import expm
import time

# =============================================================================
#  1. PARAMÈTRES DU SYSTÈME (modèle mécanique PMSM)
# =============================================================================

J   = 0.01    # [kg.m²]        Moment d'inertie rotor
f   = 0.1     # [N.m.s/rad]   Frottement visqueux
Ts  = 0.001   # [s]            Période d'échantillonnage (1 ms)

# Modèle continu d'état : scalaire (SISO)
#   dx/dt = Ac*x + Bc*u  ,  y = Cc*x
#   x = omega [rad/s],  u = Tem [N.m],  y = omega
Ac = -f / J          # = -10 rad/s
Bc =  1 / J          # =  100
Cc =  1.0

# Discrétisation exacte Zero-Order-Hold (ZOH)
Ad = float(np.exp(Ac * Ts))
Bd = float((np.exp(Ac * Ts) - 1) / Ac * Bc)
Cd = Cc

tau_m = J / f        # Constante de temps mécanique [s]

print("=" * 55)
print("  COMMANDE PRÉDICTIVE MPC — PMSM — CVXPY")
print("=" * 55)
print(f"\n  Modèle continu  : dx/dt = {Ac:.1f}·x + {Bc:.1f}·u")
print(f"  Modèle discret  : Ad = {Ad:.6f}, Bd = {Bd:.6f}")
print(f"  Ts              : {Ts*1000:.1f} ms")
print(f"  τm              : {tau_m:.3f} s\n")

# =============================================================================
#  2. PARAMÈTRES MPC
# =============================================================================

N       = 20       # Horizon de prédiction [pas]
Q_w     = 500.0    # Poids erreur de sortie   (poursuite)
R_w     = 0.01     # Poids effort de commande (douceur)

T_max   = 9.55     # [N.m]   Saturation du couple
omega_ref = 314.16 # [rad/s] Consigne = 3000 tr/min

print(f"  Horizon N       : {N} pas ({N*Ts*1000:.0f} ms)")
print(f"  Poids Q         : {Q_w}")
print(f"  Poids R         : {R_w}")
print(f"  ω_ref           : {omega_ref:.2f} rad/s")
print(f"  T_max           : {T_max} N.m\n")

# =============================================================================
#  3. DÉFINITION DU PROBLÈME MPC VIA CVXPY
#
#  À chaque pas k, on résout :
#
#    minimiser   Σ_{i=0}^{N-1} [ Q·(x_{i+1} - x_ref)² + R·u_i² ]
#
#    sujet à     x_{i+1} = Ad·x_i + Bd·u_i   (dynamique discrète)
#                |u_i|   ≤ T_max              (saturation physique)
#                x_0     = x_k               (état courant mesuré)
#
#  CVXPY construit et résout ce QP à chaque instant d'échantillonnage.
# =============================================================================

def solve_mpc(x_init, x_ref, N, Ad, Bd, Q_w, R_w, T_max):
    """
    Résout le problème MPC pour un état initial x_init.

    Retourne :
        u_opt : première commande optimale à appliquer [scalaire]
        status : statut du solveur ('optimal', etc.)
        solve_time : temps de résolution [s]
    """
    # Variables de décision CVXPY
    x = cp.Variable(N + 1)   # États prédits  : x[0], x[1], ..., x[N]
    u = cp.Variable(N)        # Commandes      : u[0], u[1], ..., u[N-1]

    # Fonction objectif
    cost = 0
    for i in range(N):
        cost += Q_w * cp.square(x[i + 1] - x_ref)   # erreur de poursuite
        cost += R_w * cp.square(u[i])                 # effort de commande

    # Contraintes
    constraints = [x[0] == x_init]                   # état initial
    for i in range(N):
        constraints += [x[i + 1] == Ad * x[i] + Bd * u[i]]   # dynamique
        constraints += [u[i] <=  T_max]              # saturation haute
        constraints += [u[i] >= -T_max]              # saturation basse

    # Résolution
    prob = cp.Problem(cp.Minimize(cost), constraints)

    t0 = time.perf_counter()
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    solve_time = time.perf_counter() - t0

    if u.value is not None:
        return float(u.value[0]), prob.status, solve_time
    else:
        # Repli : commande nulle si solveur échoue
        return 0.0, prob.status, solve_time


# =============================================================================
#  4. SIMULATION EN BOUCLE FERMÉE
# =============================================================================

t_fin  = 0.15
t_step = 0.01
t_vect = np.arange(0, t_fin + Ts, Ts)
K      = len(t_vect)

# Historiques
x_hist      = np.zeros(K)
u_hist      = np.zeros(K)
solve_times = np.zeros(K)

x_k = 0.0    # État initial : moteur à l'arrêt

print("  Simulation en cours...")
t_sim_start = time.perf_counter()

for k in range(K):
    # Consigne active après t_step
    ref_k = omega_ref if t_vect[k] >= t_step else 0.0

    # Résolution MPC
    u_k, status, st = solve_mpc(x_k, ref_k, N, Ad, Bd, Q_w, R_w, T_max)

    # Enregistrement
    x_hist[k]      = x_k
    u_hist[k]      = u_k
    solve_times[k] = st * 1000   # [ms]

    # Évolution du système (équation d'état discrète)
    x_k = Ad * x_k + Bd * u_k

total_time = time.perf_counter() - t_sim_start
print(f"  Simulation terminée en {total_time:.2f} s")
print(f"  Temps moyen par résolution CVXPY : {solve_times.mean():.2f} ms")
print(f"  Temps max par résolution         : {solve_times.max():.2f} ms\n")

# =============================================================================
#  5. CALCUL DES PERFORMANCES
# =============================================================================

k_step = np.searchsorted(t_vect, t_step)

# Temps de réponse à 5 %
seuil_95  = 0.95 * omega_ref
indices_ok = np.where(x_hist[k_step:] >= seuil_95)[0]
if len(indices_ok) > 0:
    t_rep5 = (t_vect[k_step + indices_ok[0]] - t_step) * 1000   # [ms]
else:
    t_rep5 = float('nan')

# Dépassement
overshoot = max(0.0, (np.max(x_hist) - omega_ref) / omega_ref * 100)

# Erreur statique finale
err_static = abs(omega_ref - x_hist[-1])

print("=" * 55)
print("  PERFORMANCES MPC")
print("=" * 55)
print(f"  Temps de réponse 5 %  : {t_rep5:.1f} ms")
print(f"  Dépassement           : {overshoot:.3f} %")
print(f"  Erreur statique finale: {err_static:.4f} rad/s")
print(f"  Commande max appliquée: {np.max(np.abs(u_hist)):.3f} N.m")
print("=" * 55 + "\n")

# =============================================================================
#  6. TRACÉS DES RÉSULTATS
# =============================================================================

fig = plt.figure(figsize=(13, 9), facecolor='white')
fig.suptitle(
    "MPC — Commande Prédictive PMSM | Python / CVXPY\n"
    f"N = {N} pas, Q = {Q_w}, R = {R_w} | ESP Antsiranana",
    fontsize=14, fontweight='bold', color='navy'
)
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

t_ms = t_vect * 1000   # axe temps en ms

# --- Subplot 1 : Réponse en vitesse ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ms, x_hist, 'b-', linewidth=2.5, label='ω(t) — MPC')
ax1.axhline(omega_ref, color='red', linestyle='--', linewidth=1.5,
            label=f'ω* = {omega_ref:.1f} rad/s')
ax1.axvline(t_step * 1000, color='gray', linestyle=':', linewidth=1.2, label='Échelon')
if not np.isnan(t_rep5):
    ax1.axvline(t_step * 1000 + t_rep5, color='green', linestyle='--',
                linewidth=1.2, label=f'tr5% = {t_rep5:.0f} ms', alpha=0.7)
    ax1.plot(t_step * 1000 + t_rep5, seuil_95, 'go', markersize=8,
             markerfacecolor='green', zorder=5)
ax1.set_ylabel('ω(t)  [rad/s]', fontsize=11)
ax1.set_title('Réponse indicielle en vitesse', fontsize=11)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.4)
ax1.set_ylim(-30, omega_ref * 1.15)

# --- Subplot 2 : Commande ---
ax2 = fig.add_subplot(gs[1])
ax2.step(t_ms, u_hist, 'r-', linewidth=2, where='post', label='u(t) = Tem(t)')
ax2.axhline( T_max, color='black', linestyle='--', linewidth=1,
             label=f'+T_max = {T_max} N.m', alpha=0.6)
ax2.axhline(-T_max, color='black', linestyle='--', linewidth=1,
             label=f'−T_max = {T_max} N.m', alpha=0.6)
ax2.axvline(t_step * 1000, color='gray', linestyle=':', linewidth=1.2)
ax2.set_ylabel('Tem(t)  [N.m]', fontsize=11)
ax2.set_title('Séquence de commande (couple électromagnétique)', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.4)
ax2.set_ylim(-T_max * 1.4, T_max * 1.4)

# --- Subplot 3 : Erreur de poursuite ---
ax3 = fig.add_subplot(gs[2])
ref_vec  = np.where(t_vect >= t_step, omega_ref, 0.0)
err_hist = ref_vec - x_hist
ax3.plot(t_ms, err_hist, 'm-', linewidth=2, label='ε(t) = ω* − ω(t)')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.axvline(t_step * 1000, color='gray', linestyle=':', linewidth=1.2)
ax3.set_xlabel('Temps (ms)', fontsize=11)
ax3.set_ylabel('ε(t)  [rad/s]', fontsize=11)
ax3.set_title('Erreur de poursuite', fontsize=11)
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.4)

plt.savefig('resultats_MPC.png', dpi=150, bbox_inches='tight')
print("  Figure principale sauvegardée : resultats_MPC.png")

# =============================================================================
#  7. ÉTUDE PARAMÉTRIQUE — EFFET DE L'HORIZON N
# =============================================================================

print("\n  Étude paramétrique : effet de N...")

N_vals   = [5, 10, 20, 50]
couleurs = ['blue', 'green', 'red', 'purple']

fig2, ax = plt.subplots(figsize=(11, 5), facecolor='white')

for Nv, col in zip(N_vals, couleurs):
    x_v, x_kv = np.zeros(K), 0.0
    for k in range(K):
        ref_k = omega_ref if t_vect[k] >= t_step else 0.0
        u_kv, _, _ = solve_mpc(x_kv, ref_k, Nv, Ad, Bd, Q_w, R_w, T_max)
        x_v[k] = x_kv
        x_kv   = Ad * x_kv + Bd * u_kv
    ax.plot(t_ms, x_v, color=col, linewidth=2,
            label=f'N = {Nv} pas ({Nv*Ts*1000:.0f} ms)')

ax.axhline(omega_ref, color='black', linestyle='--', linewidth=1.5,
           label=f'ω* = {omega_ref:.1f} rad/s')
ax.axvline(t_step * 1000, color='gray', linestyle=':', linewidth=1.2, label='Échelon')
ax.set_xlabel('Temps (ms)', fontsize=12)
ax.set_ylabel('ω(t)  [rad/s]', fontsize=12)
ax.set_title(f'Influence de l\'horizon de prédiction N | Q={Q_w}, R={R_w}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
ax.set_ylim(-30, omega_ref * 1.2)

fig2.savefig('etude_horizon_N.png', dpi=150, bbox_inches='tight')
print("  Figure étude N sauvegardée    : etude_horizon_N.png")
plt.show()
print("\n  Exécution terminée avec succès.")
