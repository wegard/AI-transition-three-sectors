## 📘 Model Overview: A Three-Sector Framework with Evolving Automation

This repository implements a simulation model of an economy transitioning toward near-Artificial General Intelligence (AGI). The model features three production sectors:

* **T: Traditional Sector**
* **H: Human-centric Sector**
* **I: Intelligence Sector**

The key dynamic is the **substitutability between AI capital and human labor**, evolving over time due to technological progress.

---

## ⚙️ Core Elements

### Economic Environment

* A representative household supplies a fixed labor force $L$.
* Three inputs: traditional capital $K$, AI capital $A$, and labor $L$.
* Labor can be allocated to T, H, I, or remain unemployed.

### Production Technologies

* All sectors operate under **perfect competition**.
* **T and I** use nested CES production functions that combine traditional capital, labor, and AI capital.
* **H** uses a standard CES function with only capital and labor—this sector is resistant to automation (for now).

### Capital and Labor Dynamics

* Output is allocated to investment in $K$ and $A$ based on marginal productivity.
* Capital evolves with depreciation and investment.
* Labor dynamics include:

  * **Job separation** ($\lambda_s$)
  * **Hiring from unemployment** ($\lambda_f$)
  * **Wage-based sector mobility**, governed by frictions ($\chi, \mu, \xi$)

---

## 🤖 Automation and Technology

The degree of automation in sectors T and I evolves over time following a logistic (S-curve) path:

$$
\phi_{j,t} = \phi_{j,\text{min}} + \frac{\phi_{j,\text{max}} - \phi_{j,\text{min}}}{1 + e^{-\gamma_j (t - t_{0,j})}}, \quad j \in \{T, I\}
$$

This captures the gradual but accelerating rise of AI capabilities. The human sector (H) is assumed to remain unaffected ($\phi_H = 0$).

---

## 🧮 Model Implementation

The simulation tracks:

* Sectoral outputs
* Capital accumulation
* Labor flows and unemployment
* Wages and returns
* Technological diffusion via $\phi_{T,t}$ and $\phi_{I,t}$

Investment and labor allocation decisions are responsive to marginal products and wages, respectively.

---

## 📁 Repository Structure

```text
.
├── model.py          # Core simulation code
├── parameters.yaml   # Model parameters
├── results/          # Output from simulations
├── plots/            # Generated figures
└── README.md         # Model overview (this file)
```

---

## 📜 License

This project is released under the **MIT License**—feel free to use, adapt, and build upon it with proper attribution.
