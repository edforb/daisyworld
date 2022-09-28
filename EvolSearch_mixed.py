"""
Contains the multiprocessing evolutionary search class
Madhavun Candadai
Jan, 2018
"""
# from multiprocessing import Pool
import time
import numpy as np
from pathos.multiprocessing import ProcessPool

__evolsearch_process_pool = None


class EvolSearch:
    def __init__(self, evol_params, discrete_initial_pop, continuous_initial_pop):
        """
        Initialize evolutionary search
        ARGS:
        evol_params: dict
            required keys -
                pop_size: int - population size,
                genotype_size: int - genotype_size,
                fitness_function: function - a user-defined function that takes a genotype as arg and returns a float fitness value
                elitist_fraction: float - fraction of top performing individuals to retain for next generation
                mutation_variance: float - variance of the gaussian distribution used for mutation noise
            optional keys -
                fitness_args: list-like - optional additional arguments to pass while calling fitness function
                                           list such that len(list) == 1 or len(list) == pop_size
                num_processes: int -  pool size for multiprocessing.pool.Pool - defaults to os.cpu_count()
        """
        # check for required keys
        required_keys = [
            "pop_size",
            "discrete_genotype_size",
            "continuous_genotype_size",
            "fitness_function",
            "elitist_fraction",
            "discrete_mutation_probability",
            "continuous_mutation_variance"
        ]
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception(
                    "Argument evol_params does not contain the following required key: "
                    + key
                )

        # checked for all required keys
        self.pop_size = evol_params["pop_size"]
        self.discrete_genotype_size = evol_params["discrete_genotype_size"]
        self.continuous_genotype_size = evol_params["continuous_genotype_size"]
        self.fitness_function = evol_params["fitness_function"]
        self.elitist_fraction = int(
            np.ceil(evol_params["elitist_fraction"] * self.pop_size)
        )
        self.discrete_mutation_probability = evol_params["discrete_mutation_probability"]
        self.continuous_mutation_variance = evol_params["continuous_mutation_variance"]

        # validating fitness function
        assert self.fitness_function, "Invalid fitness_function"
        rand_discrete_genotype = np.random.randint(2, size=self.discrete_genotype_size)
        rand_continuous_genotype = np.random.rand(self.continuous_genotype_size)
        rand_genotype_fitness = self.fitness_function(rand_discrete_genotype, rand_continuous_genotype)
        assert (
            type(rand_genotype_fitness) == type(0.0)
            or type(rand_genotype_fitness) in np.sctypes["float"]
        ), "Invalid return type for fitness_function. Should be float or np.dtype('np.float*')"

        # create other required data
        self.num_processes = evol_params.get("num_processes", None)
        self.discrete_pop = np.copy(discrete_initial_pop)
        self.continuous_pop = np.copy(continuous_initial_pop)
        self.fitness = np.zeros(self.pop_size)
        self.num_batches = int(self.pop_size / self.num_processes)
        self.num_remainder = int(self.pop_size % self.num_processes)

        # check for fitness function kwargs
        if "fitness_args" in evol_params.keys():
            optional_args = evol_params["fitness_args"]
            assert (
                len(optional_args) == 1 or len(optional_args) == self.pop_size
            ), "fitness args should be length 1 or pop_size."
            self.optional_args = optional_args
        else:
            self.optional_args = None

        # creating the global process pool to be used across all generations
        global __evolsearch_process_pool
        __evolsearch_process_pool = ProcessPool(self.num_processes)
        time.sleep(0.5)

    def evaluate_fitness(self, individual_index):
        """
        Call user defined fitness function and pass genotype
        """
        if self.optional_args:
            if len(self.optional_args) == 1:
                return self.fitness_function(
                    self.discrete_pop[individual_index, :], self.continuous_pop[individual_index, :], self.optional_args[0]
                )
            else:
                return self.fitness_function(
                    self.discrete_pop[individual_index, :], self.continuous_pop[individual_index, :], self.optional_args[individual_index]
                )
        else:
            return self.fitness_function(self.discrete_pop[individual_index, :], self.continuous_pop[individual_index, :])

    def elitist_selection(self):
        """
        from fitness select top performing individuals based on elitist_fraction
        """
        self.discrete_pop = self.discrete_pop[np.argsort(self.fitness)[-self.elitist_fraction :], :]
        self.continuous_pop = self.continuous_pop[np.argsort(self.fitness)[-self.elitist_fraction :], :]


    def mutation(self):
        """
        create new pop by repeating mutated copies of elitist individuals
        """
        # number of copies of elitists required
        num_reps = (
            int((self.pop_size - self.elitist_fraction) / self.elitist_fraction) + 1
        )

        # creating copies and adding noise
        mutated_discrete_elites = np.tile(self.discrete_pop, [num_reps, 1])

        replace_indices = np.random.choice([False, True], size=np.shape(mutated_discrete_elites), p=[1-self.discrete_mutation_probability, self.discrete_mutation_probability])
        
        mutated_discrete_elites[replace_indices] = np.random.randint(2, size=np.sum(replace_indices))

        mutated_continuous_elites = np.tile(self.continuous_pop, [num_reps, 1])

        mutated_continuous_elites += np.random.normal(
            loc=0.0,
            scale=self.continuous_mutation_variance,
            size=[num_reps * self.elitist_fraction, self.continuous_genotype_size],
        )

        # concatenating elites with their mutated versions
        self.discrete_pop = np.vstack((self.discrete_pop, mutated_discrete_elites))
        self.continuous_pop = np.vstack((self.continuous_pop, mutated_continuous_elites))

        # clipping to pop_size
        self.discrete_pop = self.discrete_pop[: self.pop_size, :]
        self.continuous_pop = self.continuous_pop[: self.pop_size, :]

        # clipping continuous pop to [0,1]
        self.continuous_pop = np.clip(self.continuous_pop, 0, 1)

    def step_generation(self):
        """
        evaluate fitness of pop, and create new pop after elitist_selection and mutation
        """
        global __evolsearch_process_pool

        if not np.all(self.fitness == 0):
            # elitist_selection
            self.elitist_selection()

            # mutation
            self.mutation()

        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(
                    self.evaluate_fitness, np.arange(self.pop_size)
                )
            )
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(
                    self.evaluate_fitness, np.arange(self.pop_size)
                )
            )

    def execute_search(self, num_gens):
        """
        runs the evolutionary algorithm for given number of generations, num_gens
        """
        # step generation num_gens times
        for gen in np.arange(num_gens):
            self.step_generation()

    def get_fitnesses(self):
        """
        simply return all fitness values of current population
        """
        return self.fitness

    def get_best_individual(self):
        """
        returns 1D array of the genotype that has max fitness
        """
        best_individual_index = np.argmax(self.fitness)
        return self.discrete_pop[best_individual_index, :], self.continuous_pop[best_individual_index, :]

    def get_best_individual_fitness(self):
        """
        return the fitness value of the best individual
        """
        return np.max(self.fitness)

    def get_mean_fitness(self):
        """
        returns the mean fitness of the population
        """
        return np.mean(self.fitness)

    def get_fitness_variance(self):
        """
        returns variance of the population's fitness
        """
        return np.std(self.fitness) ** 2