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
import daisyEvo as evo
import networkx as nx

### PARAMETERS ##

# Temperatures
KELVIN_OFFSET = 273.15
Td_min = 5 + KELVIN_OFFSET
Td_max = 40 + KELVIN_OFFSET

# Flux terms
So = 1000
sigma = 5.67032e-8

# Convergence
maxconv = 100

# Bio Info


# Perturbation
perturbation = 0
pert_value = 1.1

display = True


### CLASSES ###

class world:
    def __init__(self, genotype, diversity):
        # Convergence Criteria
        self.diversity = diversity
        self.alb_low = 0.1
        self.alb_high = 0.9
        self.alb_int = (self.alb_high-self.alb_low)/self.diversity
        self.alb_barren = 0.4
        self.maxconv = maxconv
        self.tol = 0.000001
        self.Sflux_min = 0.5
        self.Sflux_max = 2.0
        self.Sflux_step = 0.002
        self.fluxes = np.arange(self.Sflux_min, self.Sflux_max, self.Sflux_step)
        self.insul = 20
        self.spec_list = []
        self.web = np.reshape(genotype, (diversity,diversity))
        np.fill_diagonal(self.web,0)
        self.init_life = 0
        self.end_life = 0
        self.duration = 0
        
class Species:
    def __init__(self, world, troph, alb):
        self.troph = troph
        self.area = 0.01
        self.min_area = 0.01
        self.area_vec = np.zeros_like(world.fluxes)
        self.alb = alb
        self.Td_ideal = 22.5 + KELVIN_OFFSET
        self.Td = 0
        self.dA = 2*world.tol
        self.darea = 0
        self.darea_old = 0
        self.birth = 0
        self.drate = 0.3

### SIMULATION ###

def daisysimulation(genotype, diversity):
    if __name__ == '__main__':
        """Run the daisyworld model"""
        # Initialize arrays
        
        daisyworld = world(genotype, diversity)
        
        counter_troph = 0
        counter_alb = daisyworld.alb_low
        for i in daisyworld.web[0]:
            d = Species(daisyworld,counter_troph,counter_alb)
            daisyworld.spec_list.append(d)
            counter_alb = counter_alb + daisyworld.alb_int
            counter_troph = counter_troph + 1
        
        area_vec = np.zeros_like(daisyworld.fluxes)
        area_barren_vec = np.zeros_like(daisyworld.fluxes)
        
        Tp_vec = np.zeros_like(daisyworld.fluxes)
    
        # Loop over fluxes
        for j, flux in enumerate(daisyworld.fluxes):
            
            #if flux == pert_value:
                #flux = flux + perturbation
            
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
                s.dA = 2*daisyworld.tol
                s.darea_old = 0
            min_dA = 2*daisyworld.tol
            ## SET NEW MIN AT END
    
            while it <= daisyworld.maxconv and min_dA > daisyworld.tol:
                
                # Planetary albedo
                alb_p = 0
                for s in daisyworld.spec_list:
                    alb_p = alb_p + s.area*s.alb
                alb_p = alb_p + area_barren*daisyworld.alb_barren
                
                ## PLANETARY ALBEDO KEEPS GOING OVER 1
                # print(alb_p)
                
                # Planetary temperature
                Tp = np.power(flux*So*(1-alb_p)/sigma, 0.25)
                
                # Local temperatures
                for s in daisyworld.spec_list:
                    s.Td = daisyworld.insul*(alb_p-s.alb) + Tp
                
                # Determine birth rates
                ### THIS THE INDIVIDUAL SPECIES RELATIONSHIP WITH TEMPERATURE
                for s in daisyworld.spec_list:
                    if (s.Td >= Td_min
                        and s.Td <= Td_max
                        and s.area >= 0.005):
                        s.birth = 1 - 0.003265*(s.Td-s.Td_ideal)**2
                    else:
                        s.birth = 0.0
        
                # Change in areal extents
                ### THIS THE ECOLOGICAL INTERACTION EQUATIONS
                
                
                for s in daisyworld.spec_list:
                    
                    ID = s.troph
                    predator = daisyworld.web[ID]
                    prey = daisyworld.web[:,ID]
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
                    
                    area_change = base_growth - s.drate + sum(predator_factor) - sum(prey_factor)
                    s.darea = s.area * area_change
                    
                # Change from previous iteration
                for s in daisyworld.spec_list:
                    s.dA = abs(s.darea - s.darea_old)
    
                # Update areas, states, and iteration count
                # Use small number to have change happen slowly
                area_counter_2 = 0
                for s in daisyworld.spec_list:                        
                    #s.darea_old = (1/50)*s.darea
                    s.darea_old = (1/50)*s.darea
                    #s.area = s.area + (1/50)*s.darea
                    s.area = s.area + (1/50)*s.darea
                    area_counter_2 = area_counter_2 + s.area
                area_barren = 1-area_counter_2
                it += 1
        
            # Save states
            area_counter_3 = 0
            for s in daisyworld.spec_list:
                s.area_vec[j] = s.area
                area_counter_3 = area_counter_3 + s.area
                    
            area_vec[j] = area_counter_3
            area_barren_vec[j] = area_barren
            Tp_vec[j] = Tp
                
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
            if daisyworld.end_life != 0:
                break
        
            ### GRAPHS ###
        if display == True:   
            fig, ax = plt.subplots(2, 1)
                    
            for s in daisyworld.spec_list:
                ax[0].plot(daisyworld.fluxes, 100*s.area_vec, color='gray', label='black')
            ax[0].plot(daisyworld.fluxes, 100*area_vec, color='black', label='total')
            ax[0].plot(daisyworld.fluxes, 100*area_barren_vec, color='brown', label='total')
            
            ax[0].set_xlabel('Solar Luminosity')
            ax[0].set_ylabel('Coverage Area (%)')
            
                
            ax[1].plot(daisyworld.fluxes, Tp_vec-KELVIN_OFFSET, color='red')
            ax[1].set_xlabel('Solar Luminosity')
            ax[1].set_ylabel('Global Temperature (C)')
            plt.show()
            
        daisyworld.duration = daisyworld.end_life - daisyworld.init_life
                
        print(daisyworld.duration)
        return daisyworld
        



######################
## EVOLUTIONARY STUFF
######################

###  General parameters
diversity = 3
popsize = 100
recombProb = 0.75
mutatProb = 0.1
generations = 5


def fitnessFunction(genotype):
    sim = daisysimulation(genotype, diversity)
    return sim.duration

###  Microbial test
demeSize = 2
ga = evo.Microbial(fitnessFunction, popsize, diversity, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.showFitness()
ga.save("microbialresults")


######################
## NETWORK DISPLAY
######################

bestind = ga.pop[np.argmax(ga.fitness)]
bestind = np.reshape(bestind, (diversity,diversity))

G = nx.DiGraph(bestind)
nx.draw(G, with_labels=True, font_weight='bold')












