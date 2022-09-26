import pickle
import numpy as np
from EvoDaisy import daisyworld_fitness

with open("best_individual", "rb") as f:
    best_individual = pickle.load(f)
    
daisyworld_fitness(best_individual["params"], diversity=30, display=True, maxconv=100, fluxes=np.arange(0.5, 3.0, 0.02), pert_value=[0], perturbation=0)
    