#!/usr/bin/env python

from tkinter import W
import numpy as np
import pickle
# WARNING I AM FILTERING WARNINGS BECUASE PATHOS DOESN'T LIKE THEM
import warnings
warnings.filterwarnings("ignore")

from EvolSearch import EvolSearch
from EvoDaisy import daisyworld_fitness
from functools import partial


use_best_individual = False
if use_best_individual:
    with open("best_individual", "rb") as f:
        best_individual = pickle.load(f)

########################
# Parameters
########################

diversity = 30
display = False
maxconv = 100

####### WORLD INPUT ########
## Steady Increase
#fluxes = np.arange(0.3, 1.3, 0.02)

## Seasons
#fluxes = (np.sin(np.arange(1,25,0.1))+4.5)/6
fluxes = np.arange(0.5, 3.0, 0.02)

####### PERTURBATION ########
# Perturbation
perturbation = -1.0
pert_value = list(range(125,150))

## Noise
add_noise = True

if add_noise == True:
    noise = (-0.5 + np.random.sample(len(fluxes)))/10
    fluxes = fluxes + noise


########################
# Evolve Solutions
########################

pop_size = 200
genotype_size = diversity * diversity

evol_params = {
    "num_processes": 100,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": partial(daisyworld_fitness, diversity=diversity, display=display, 
                                maxconv=maxconv, fluxes=fluxes, pert_value=pert_value,
                                perturbation=perturbation),  # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generation
    "mutation_variance": 0.1,  # mutation noise added to offspring.
}
initial_pop = np.random.randint(2, size=(pop_size, diversity*diversity))

if use_best_individual:
    initial_pop[0] = best_individual["params"]

evolution = EvolSearch(evol_params, initial_pop)

save_best_individual = {
    "params": None,
    "diversity": diversity,
    "maxconv": maxconv,
    "best_fitness": [],
    "mean_fitness": [],
}

for i in range(20):
    evolution.step_generation()
    
    save_best_individual["params"] = evolution.get_best_individual()
    
    save_best_individual["best_fitness"].append(evolution.get_best_individual_fitness())
    save_best_individual["mean_fitness"].append(evolution.get_mean_fitness())

    print(
        len(save_best_individual["best_fitness"]), 
        save_best_individual["best_fitness"][-1], 
        save_best_individual["mean_fitness"][-1]
    )

    with open("best_individual", "wb") as f:
        pickle.dump(save_best_individual, f)