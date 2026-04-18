# Galaxy Morpho

Galaxy Morpho is a project on galaxy morphology classification and galaxy formation simulation. It combines theoretical astrophysics, machine learning, and N-body simulation to study how different galaxy structures form and evolve.

## Overview

The project is divided into three parts:

### Module 1 – Galaxy Formation Theory
This section explores the physical processes behind galaxy formation, including:
- Dark matter halos
- Gas cooling and baryonic physics
- Stellar evolution
- Feedback processes
- Hierarchical mergers
- Environmental effects

It also includes Newtonian analysis showing why galaxies tend to flatten into rotating disk structures.

### Module 2 – Galaxy Morphology Classification
This module uses machine learning to classify galaxies into spiral and elliptical types using unlabeled astronomical data.

Main steps:
- Feature engineering
- Data splitting by sky region
- Neural network embeddings
- KMeans clustering
- Pseudo-label generation
- Supervised classification on confident samples

### Module 3 – Galaxy Formation Simulation
This module simulates galaxy formation using an N-body model.

By changing the initial angular momentum:
- High angular momentum → Spiral galaxy
- Low angular momentum → Elliptical galaxy
- Intermediate/random angular momentum → Irregular galaxy

The simulation uses:
- Newtonian gravity
- Euler integration
- Pygame for animation
- Matplotlib for energy and angular momentum plots

---

## Repository Structure

```text
Galaxy-Morpho/
│
├── README.md
├── project_report.pdf
├── module2_classification.ipynb
└── module3_simulation.py
