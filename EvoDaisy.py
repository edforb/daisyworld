#!/usr/bin/env python

### BASED ON LOVELOCK 92 MODEL EXTENSIONS ###

"""
Evolutionary version of the Daisyworld model described in:

    Watson, A.J.; Lovelock, J.E (1983). "Biological homeostasis of
    the global environment: the parable of Daisyworld". Tellus.35B:
    286–9.

by Eden Forbes
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import statistics


def daisyworld_fitness(genotype, diversity, maxconv, display, fluxes, pert_value, perturbation):
    """Run the daisyworld model"""

    class world:
        def __init__(self, genotype, diversity, fluxes):
            # Convergence Criteria
            self.diversity = diversity
            self.alb_low = 0.1
            self.alb_high = 0.9
            self.alb_int = (self.alb_high - self.alb_low) / self.diversity
            self.alb_barren = 0.4
            self.maxconv = maxconv
            self.tol = 0.000001
            self.Sflux_min = 0.5
            self.Sflux_max = 2.0
            self.Sflux_step = 0.002
            self.fluxes = fluxes
            self.insul = 20
            self.spec_list = []
            self.web = np.reshape(genotype, (diversity, diversity))
            np.fill_diagonal(self.web, 0)
            self.init_life = 0
            self.end_life = 0
            self.duration = 0
            self.pert_value = pert_value
            self.perturbation = perturbation

    class Species:
        def __init__(self, world, troph, alb):
            self.troph = troph
            self.area = 0.01
            self.min_area = 0.01
            self.area_vec = np.zeros_like(world.fluxes)
            self.alb = alb
            self.Td_ideal = 22.5 + KELVIN_OFFSET
            self.Td = 0
            self.dA = 2 * world.tol
            self.darea = 0
            self.darea_old = 0
            self.birth = 0
            self.drate = 0.3

    # Temperatures
    KELVIN_OFFSET = 273.15
    Td_min = 5 + KELVIN_OFFSET
    Td_max = 40 + KELVIN_OFFSET

    # Flux terms
    So = 1000
    sigma = 5.67032e-8

    # Initialize arrays

    daisyworld = world(genotype, diversity, fluxes)

    counter_troph = 0
    counter_alb = daisyworld.alb_low
    for i in daisyworld.web[0]:
        d = Species(daisyworld, counter_troph, counter_alb)
        daisyworld.spec_list.append(d)
        counter_alb = counter_alb + daisyworld.alb_int
        counter_troph = counter_troph + 1

    area_vec = np.zeros_like(daisyworld.fluxes)
    area_barren_vec = np.zeros_like(daisyworld.fluxes)

    Tp_vec = np.zeros_like(daisyworld.fluxes)
    Tp_dead_vec = np.zeros_like(daisyworld.fluxes)

    # Loop over fluxes
    for j, flux in enumerate(daisyworld.fluxes):

        if j in daisyworld.pert_value:
            flux = flux + daisyworld.perturbation

        # Minimum species coverage
        area_covered = 0
        for s in daisyworld.spec_list:
            if s.area < s.min_area:
                s.area = s.min_area
            area_covered = area_covered + s.area
        area_barren = 1 - area_covered

        # Reset iteration metrics
        it = 0
        for s in daisyworld.spec_list:
            s.dA = 2 * daisyworld.tol
            s.darea_old = 0
        min_dA = 2 * daisyworld.tol
        ## SET NEW MIN AT END

        while it <= daisyworld.maxconv and min_dA > daisyworld.tol:

            # Planetary albedo
            alb_p = 0
            for s in daisyworld.spec_list:
                alb_p = alb_p + s.area * s.alb
            alb_p = alb_p + area_barren * daisyworld.alb_barren

            ## PLANETARY ALBEDO KEEPS GOING OVER 1
            # print(alb_p)

            # Planetary temperature
            Tp = np.power(flux * So * (1 - alb_p) / sigma, 0.25)
            Tp_dead = np.power(flux * So * (1 - daisyworld.alb_barren) / sigma, 0.25)

            # Local temperatures
            for s in daisyworld.spec_list:
                s.Td = daisyworld.insul * (alb_p - s.alb) + Tp

            # Determine birth rates
            ### THIS THE INDIVIDUAL SPECIES RELATIONSHIP WITH TEMPERATURE
            for s in daisyworld.spec_list:
                if s.Td >= Td_min and s.Td <= Td_max and s.area >= 0.005:
                    s.birth = 1 - 0.003265 * (s.Td - s.Td_ideal) ** 2
                else:
                    s.birth = 0.0

            # Change in areal extents
            ### THIS THE ECOLOGICAL INTERACTION EQUATIONS

            for s in daisyworld.spec_list:

                ID = s.troph
                predator = daisyworld.web[ID]
                prey = daisyworld.web[:, ID]
                predator_factor = []
                prey_factor = []
                base_growth = s.birth * area_barren
                pred_count = 0
                for p in predator:
                    if p == 1:
                        predator_factor.append(daisyworld.spec_list[pred_count].area)
                    pred_count = pred_count + 1
                prey_count = 0
                for p in prey:
                    if p == 1:
                        prey_factor.append(daisyworld.spec_list[prey_count].area)
                    prey_count = prey_count + 1

                area_change = (
                    base_growth - s.drate + sum(predator_factor) - sum(prey_factor)
                )
                s.darea = s.area * area_change

            # Change from previous iteration
            for s in daisyworld.spec_list:
                s.dA = abs(s.darea - s.darea_old)

            # Update areas, states, and iteration count
            # Use small number to have change happen slowly
            area_counter_2 = 0
            for s in daisyworld.spec_list:
                # s.darea_old = (1/50)*s.darea
                s.darea_old = (1 / 50) * s.darea
                # s.area = s.area + (1/50)*s.darea
                s.area = s.area + (1 / 50) * s.darea
                area_counter_2 = area_counter_2 + s.area
            area_barren = 1 - area_counter_2
            it += 1

        # Save states
        area_counter_3 = 0
        for s in daisyworld.spec_list:
            s.area_vec[j] = s.area
            area_counter_3 = area_counter_3 + s.area

        area_vec[j] = area_counter_3
        area_barren_vec[j] = area_barren
        Tp_vec[j] = Tp
        Tp_dead_vec[j] = Tp_dead

        # Check life init, end
        current_max = 0
        for s in daisyworld.spec_list:
            if s.area > current_max:
                current_max = s.area
        if daisyworld.init_life == 0:
            if current_max > daisyworld.spec_list[0].min_area:
                daisyworld.init_life = flux
        if daisyworld.init_life != 0:
            if current_max < daisyworld.spec_list[0].min_area:
                daisyworld.end_life = flux
        #if daisyworld.end_life != 0:
         #   break
        
    if daisyworld.end_life == 0:
        daisyworld.end_life = daisyworld.fluxes[-1]

        ### GRAPHS ###
    if display == True:
        fig, ax = plt.subplots(2, 1)

        for s in daisyworld.spec_list:
            ax[0].plot(list(range(len(daisyworld.fluxes))), 100 * s.area_vec, label=s.troph)
       # ax[0].plot(list(range(len(daisyworld.fluxes))), 100 * area_vec, color="black", label="total")
       # ax[0].plot(
       #     list(range(len(daisyworld.fluxes))), 100 * area_barren_vec, color="brown", label="total"
       # )

        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Coverage Area (%)")

        ax[1].plot(list(range(len(daisyworld.fluxes))), Tp_vec - KELVIN_OFFSET, color="red")
        ax[1].plot(list(range(len(daisyworld.fluxes))), Tp_dead_vec - KELVIN_OFFSET, color="gray")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Global Temperature (C)")
        plt.show()

    daisyworld.duration = daisyworld.end_life - daisyworld.init_life
    
    ## CHOOSE FITNESS FUNCTION
    return statistics.mean(area_vec)
    # return daisyworld.duration

