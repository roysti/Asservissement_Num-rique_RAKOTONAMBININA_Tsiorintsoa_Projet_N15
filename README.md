# Asservissement_Numérique_RAMAHANDRY_Tsiory_Projet1

## Commande Prédictive Simple (MPC) sans Toolbox — Python / CVXPY

**Auteur :** RAKOTONAMBININA Jehovamiahy Tsiorintsoa  
**Filière :** M2 Mécatronique — École Supérieure Polytechnique d'Antsiranana (ESP)  
**Cours :** Asservissement Numérique  
**Année :** 2025–2026  

---

## 📌 Description du projet

Ce projet implémente une **commande prédictive (MPC — Model Predictive Control)** appliquée à la régulation de vitesse d'un Moteur Synchrone à Aimants Permanents (PMSM), sans utiliser aucune toolbox MPC dédiée.

Le problème d'optimisation est formulé et résolu à chaque pas d'échantillonnage via la bibliothèque **CVXPY**, qui offre une interface claire pour les problèmes de programmation quadratique (QP) convexe.

---

## ⚙️ Principe du MPC

À chaque instant d'échantillonnage **k**, le régulateur :

1. **Prédit** l'évolution future du système sur un horizon de **N pas** :
   ```
   x[k+i+1] = Ad·x[k+i] + Bd·u[k+i]
   ```

2. **Résout** le problème d'optimisation quadratique :
   ```
   min  Σ [ Q·(x[i+1] - ω*)² + R·u[i]² ]
   
   s.c. x[i+1] = Ad·x[i] + Bd·u[i]    (dynamique)
        |u[i]| ≤ T_max                  (saturation)
        x[0]   = x_k                    (état mesuré)
   ```

3. **Applique** uniquement la première commande optimale **u[0]**, puis recommence.

---

## 🔧 Système commandé

Modèle mécanique du PMSM (du 1er ordre) :

```
G(s) = Ω(s) / Tem(s) = 1 / (J·s + f)
```

| Paramètre | Symbole | Valeur | Unité |
|-----------|---------|--------|-------|
| Inertie rotor | J | 0,01 | kg·m² |
| Frottement visqueux | f | 0,1 | N·m·s/rad |
| Constante de temps | τm = J/f | 0,1 | s |
| Couple max (saturation) | T_max | 9,55 | N·m |
| Consigne de vitesse | ω* | 314,16 | rad/s |
| Période d'échantillonnage | Ts | 1 | ms |

---


---

## 🚀 Instructions d'exécution

### 1. Prérequis

Python 3.8+ avec les bibliothèques suivantes :

```bash
pip install cvxpy numpy matplotlib scipy
```

> Le solveur **OSQP** est inclus automatiquement avec CVXPY.

### 2. Lancer le script

```bash
python MPC_commande_predictive.py
```

### 3. Résultats attendus dans la console

```
=======================================================
  COMMANDE PRÉDICTIVE MPC — PMSM — CVXPY
=======================================================
  Modèle discret  : Ad = 0.990005, Bd = 0.099005
  Ts              : 1.0 ms
  Horizon N       : 20 pas (20 ms)
  ...
  PERFORMANCES MPC
  Temps de réponse 5 % : ~30 ms
  Dépassement          : 0.000 %
  Erreur statique      : ~0.0001 rad/s
```

### 4. Figures générées

- **`resultats_MPC.png`** — 3 sous-graphes : réponse vitesse, commande, erreur
- **`etude_horizon_N.png`** — Comparaison pour N = 5, 10, 20, 50 pas

---

## 📊 Paramètres MPC (ajustables)

| Paramètre | Variable | Valeur par défaut | Effet |
|-----------|----------|-------------------|-------|
| Horizon de prédiction | `N` | 20 | Plus grand → plus stable, plus lent à calculer |
| Poids erreur | `Q_w` | 500 | Plus grand → poursuite plus agressive |
| Poids commande | `R_w` | 0.01 | Plus grand → commande plus douce |

---

## 📚 Références

1. J. M. Maciejowski, *Predictive Control with Constraints*, Prentice Hall, 2002.
2. S. J. Qin and T. A. Badgwell, "A survey of industrial model predictive control technology," *Control Engineering Practice*, vol. 11, pp. 733–764, 2003.
3. CVXPY Documentation — [https://www.cvxpy.org](https://www.cvxpy.org)
4. R. Krishnan, *Permanent Magnet Synchronous and Brushless DC Motor Drives*, CRC Press, 2010.


