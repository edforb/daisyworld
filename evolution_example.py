import matplotlib.pyplot as plt
import numpy as np

from CTRNN import CTRNN
from EvolSearch import EvolSearch
from fitnessFunction_vehicle import fitnessFunction_vehicle

import pickle

# WARNING I AM FILTERING WARNINGS BECUASE PATHOS DOESN'T LIKE THEM
import warnings

warnings.filterwarnings("ignore")

use_best_individual = False
with open("best_individual2", "rb") as f:
   best_individual = pickle.load(f)

########################
# Parameters
########################

ctrnn_size = 50
ctrnn_step_size = 0.01
transient_steps = 100
discrete = True

bv_step_size = 0.05
bv_duration = 20
bv_distance = 5


########################
# Evolve Solutions
########################

pop_size = 1000
genotype_size = ctrnn_size ** 2 + 2 * ctrnn_size


evol_params = {
    "num_processes": 200,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": lambda x: fitnessFunction_vehicle(
        x, ctrnn_size, ctrnn_step_size, bv_duration, bv_distance, bv_step_size, transient_steps, discrete=discrete
    ),  # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generation
    "mutation_variance": 0.05,  # mutation noise added to offspring.
}
initial_pop = np.random.uniform(size=(pop_size, genotype_size))
if use_best_individual:
    initial_pop[0] = best_individual["params"]

evolution = EvolSearch(evol_params, initial_pop)

save_best_individual = {
   "params": None,
   "discrete": discrete,
   "ctrnn_size": ctrnn_size,
   "ctrnn_step_size": ctrnn_step_size,
   "bv_step_size": bv_step_size,
   "bv_duration": bv_duration,
   "bv_distance": bv_distance,
   "transient_steps": transient_steps,
   "best_fitness": [],
   "mean_fitness": [],
}

for i in range(100):
    evolution.step_generation()
    
    save_best_individual["params"] = evolution.get_best_individual()
    
    save_best_individual["best_fitness"].append(evolution.get_best_individual_fitness())
    save_best_individual["mean_fitness"].append(evolution.get_mean_fitness())

    print(
        len(save_best_individual["best_fitness"]), 
        save_best_individual["best_fitness"][-1], 
        save_best_individual["mean_fitness"][-1]
    )

    with open("best_individual4", "wb") as f:
        pickle.dump(save_best_individual, f)