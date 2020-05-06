from lib.stats import *
from tqdm import tqdm
import pandas as pd
from datetime import timedelta, datetime, date
import numpy as np
import math
import pickle
import time
import os
import copy

# Some tunable simulation parameters
EE_SPREAD_SIM_DAYS_BACK = 10  # Need to run every day to get fresh data

EE_SIM_DAYS_TO_SIMULATE = 90
EE_SIM_NUM_SIMULATIONS = 1000 # How many simulations for Monte-Carlo?
EE_MOBILITY_DATA_FILENAME = "Area_OD_matrix.pkl"

# This parameter is rather sensitive. It was tuned considering the magnitudes of the probabilities of traveling
# to given regions from given region. Feel free to modify it as needed.
EE_FRACTION_STAYING_ACTIVE = 0.005  # Fraction of those staying active when considering travel probability

# Some additonal probabilities related to mobility.
# Changing these will allow to simulate the effects of region-based lockdowns
EE_PROB_MOBILITY_BETWEEN_REGIONS_IS_UNRESTRICTED = 1.0
EE_PROB_MOBILITY_BETWEEN_SAAREMAA_AND_MAINLAND_IS_UNRESTRICTED = 1.0

# Additional check "will infect" when generating the population based on statistical data
EE_P_ALREADY_INFECTED_INFECT_INITIAL = 0.1
EE_WHEN_INITIAL_POP_INFECT_SOMEONE_FACTOR = 0.8  # Related to generation interval

# NB! A modified SEIR model is used for simulation

### Necessary objects for simulation

# At the moment, for "Person" we consider only 3 states: E, I and R
# Assumption: the person will infect someone only after E->I and
# according to probability P(inf) which is P(inf)=R0 for the area
class Person:

    def __init__(self, id, area_id, t, simul, gen_correction=1):

        self.person_id = id

        # This is a reference to the simulation. This is needed to fetch some global parameters
        self.simul = simul

        self.covid19 = Covid19()  # Sorry, you have it now
        self.state = "E"  # Default state is exposed
        self.area_id = area_id

        self.timestamp = t  # Simulation time in days (0 = first day, 1 = second, etc.)

        # When will the incubation period end?
        self.symptoms_onset = self.covid19.get_days_to_symptoms() + t

        # These will be updated in E->I
        self.recover_time = None

        # Assign infection list depending on R0 in the region.
        self.will_infect_list = []
        self.create_will_infect_list_random_areas(t, gen_correction=gen_correction)  # Initialize the list

    def create_will_infect_list_random_areas(self, t=0, gen_correction=1):

        # How many people will this person infect?
        N_infects = 0

        r0 = self.simul.r0_per_region[self.area_id]
        if r0 > 1:
            N_infects += int(math.modf(r0)[1])
            r0 = math.modf(r0)[0]

        # Check against leftover R0
        N_infects += 1 if np.random.random() <= r0 else 0

        # Now, create a list with infection dates and areas
        inf_list = []
        for k in range(N_infects):
            inf_when = int(gen_correction * self.covid19.get_generation_interval()) + t
            inf_where, restrict = self.get_infection_area()
            inf_list.append((inf_when, inf_where, restrict))

        # Store the list
        self.will_infect_list = inf_list

    # Exposed to infected (time = simulation time)
    def check_state_e_to_i(self, t, logfn=None):

        # We count recovery time from this moment
        if self.state == "E" and self.symptoms_onset == t:

            # The person is now infected and showing symptoms
            self.state = "I"

            # When will the person recover?
            days_to_recover = self.covid19.get_days_to_recovery()
            self.recover_time = days_to_recover + t

            # If log redirect exists, log this
            if logfn is not None and self.simul.do_log:
                logfn("Day " + str(t) + ": Person " + str(self.person_id) + " in area " + str(self.area_id) +
                                        " begins showing symptoms and will recover in " + str(days_to_recover) +
                                        " days")

    # This needs an AREAS argument which is the dictionary of lists that contain all the areas with E/I/R persons
    # (from Simulation structure)
    def check_infect_someone(self, t, logfn=None):

        areadict = self.simul.pool

        # Figure out which of the infections are taking place today
        if len(self.will_infect_list) > 0:
            will_infect_today = [i for i in self.will_infect_list if t >= i[0]]
            self.will_infect_list = [i for i in self.will_infect_list if t < i[0]]  # Update the list

            # Proceed with infecting people
            for i in will_infect_today:
                if areadict[i[1]][0] >= len(areadict[i[1]][1]):
                    id_to_infect = self.simul.next_person_id
                    npc = Person(id_to_infect, i[1], t, self.simul)
                    self.simul.next_person_id += 1  # Need to keep track of this
                    areadict[i[1]][1].append(npc)

                    if logfn is not None and self.simul.do_log:
                        logfn("Day " + str(t) + ": " + ("[Mobility restricted] " if i[2] else "") +
                              "Person " + str(self.person_id) + " from area " + str(self.area_id) +
                              " infects a new person " + str(id_to_infect) + " in area " +
                              str(i[1]))

    # Infected to recovered (time = simulation time)
    def check_state_i_to_r(self, t, logfn=None):

        # The person has recovered and is moved out of the active cases pool
        if self.state == "I" and self.recover_time == t:
            self.state = "R"
            if logfn is not None and self.simul.do_log:
                logfn("Day " + str(t) + ": Person " + str(self.person_id) + " in area " +
                                        str(self.area_id) + " is moved to recovered pool")

    def get_infection_area(self):

        # Compute the probability of the person infecting another in a different area
        md = self.simul.mobility_dict

        # Stay in own area: a fraction of people remaining in their own area with a chance of interaction
        pop_in_this_area = self.simul.pop_sizes[self.area_id]
        stay_act = self.simul.fraction_stay_active * pop_in_this_area

        travel_list = []  # Note that the sequence is important here
        for i in range(8):
            new_area_id = i+1
            if new_area_id != self.area_id:
                travel_prob = np.random.normal(md[self.area_id][new_area_id][0], md[self.area_id][new_area_id][1])
                travel_list.append(travel_prob if travel_prob > 0 else 0)  # Add zero for negative values
            else:
                travel_list.append(stay_act/pop_in_this_area)

        # Get actual figures
        travel_list = [(p * pop_in_this_area) for p in travel_list]

        # Probabilities of travel to other places
        trls = sum(travel_list)
        p_travel = [ab/trls for ab in travel_list]

        # print("For area", self.area_id, "the probabilities to travel to other areas are") # Debug
        # print(p_travel) # Debug

        chance_area = np.random.choice(np.arange(1,9), p=p_travel)

        # Check against travel restriction probabilities
        this_mobility_restricted = False

        # Check mobility to and from Saaremaa
        if self.area_id == 6 or chance_area == 6:
            # Check restriction probability
            if not (np.random.random() <= self.simul.prob_mob_saare_reg_unrestricted):
                this_mobility_restricted = True
        # Or check inter-region mobility
        elif self.area_id != chance_area:
            if not (np.random.random() <= self.simul.prob_mob_regions_unrestricted):
                this_mobility_restricted = True

        # If mobility is restricted, assign this restriction to person and return his home area id
        if this_mobility_restricted:
            chance_area = self.area_id

        # Return the most likely infection area
        return chance_area, this_mobility_restricted


# The simulation class. Handles a single simulation
class Covid19SimulationEEV1:

    def __init__(self, data_url=None, log_redirect=None, gen_params=None):

        # Simulation timestamp
        self.created_timestamp = datetime.now()

        # Logging enabled by default
        self.do_log = True

        # Population sizes
        self.pop_sizes = ee_covid19_area_population_2016()

        # Check initial infected population generation parameters

        # Store the gen_params for future use
        self.gen_params = gen_params  # If None, will ignore later

        # Back N days
        days_back = EE_SPREAD_SIM_DAYS_BACK
        if gen_params is not None and "days_back" in gen_params:
            days_back = gen_params["days_back"]

        # Generation interval fraction for the initial pool of infected
        self.init_gen_interval_factor = EE_WHEN_INITIAL_POP_INFECT_SOMEONE_FACTOR
        if gen_params is not None and "gen_interval_factor" in gen_params:
            self.init_gen_interval_factor = gen_params["gen_interval_factor"]

        # Probability that the members of the initial pool of infected will infect someone
        self.prob_initial_infect_another = EE_P_ALREADY_INFECTED_INFECT_INITIAL
        if gen_params is not None and "prob_will_infect" in gen_params:
            self.prob_initial_infect_another = gen_params["prob_will_infect"]

        self.next_person_id = 1
        self.initial_person_last_id = None  # Will be used to reset the person_id on every run
        self.time = 0  # Timestamp. We need not use dates since we can guarantee that the simulation step is 1 day
        self.stop_time = EE_SIM_DAYS_TO_SIMULATE  # For how many days to we simulate the spread?
        self.days_back = timedelta(days=days_back) # Days back from now we finish initialization
        self.simulation_performed = datetime.now()

        # Infection parameters
        self.r0_per_region = EE_COVID19_area_R0  # This can be changed, expectsa dict with 8 entries for each region

        # Mobility related parameters
        self.fraction_stay_active = EE_FRACTION_STAYING_ACTIVE
        self.prob_mob_regions_unrestricted = EE_PROB_MOBILITY_BETWEEN_REGIONS_IS_UNRESTRICTED
        self.prob_mob_saare_reg_unrestricted = EE_PROB_MOBILITY_BETWEEN_SAAREMAA_AND_MAINLAND_IS_UNRESTRICTED

        if data_url is None:
            data_url = EE_URL_COVID19_TEST_RESULTS_PER_COUNTY

        self.mobility_dict = None  # NB! Need to load this one to get information about mobility between areas
        self.load_mobility_data("data" + os.sep + EE_MOBILITY_DATA_FILENAME)

        self.pool = None  # This is used on every simulation run

        # Generate initial pool of infected
        self.date_start = None
        self.initial_pool = None
        self.initial_pool_aux_data = None
        self.generate_infected_pool(data_url, log_redirect=log_redirect)

        # Run parameters
        self.simulation_running = False
        self.simulation_abort = False

    def initial_pool_generate_person(self, p):
        # NB! Assumption: we cut generation intervals by a factor in the initial infected population
        # because there is no data about when the person was exposed and the generation interval is counted from thence
        pers = Person(self.next_person_id, p.area_id, 0, self, gen_correction=self.init_gen_interval_factor)

        pers.state = "I"
        pers.recover_time = p.days_to_recovery_from_now

        # Remove the infection list from this person if chance says so
        if not (np.random.random() <= self.prob_initial_infect_another):
            pers.will_infect_list = []

        self.next_person_id += 1

        return pers

    # We need a list of person of type "InfectedPerson" here that is generated by parsing the file by a stats function
    def generate_infected_pool(self, data_url, log_redirect=None):

        initial_pool = {}
        popmaxsizes = ee_covid19_area_population_2016()

        initial_situation, auxdata = \
            ee_parse_infected_dynamically_and_assign_to_areas_until_date(self.simulation_performed - self.days_back,
                                                                         log_redirect_fn=log_redirect,
                                                                         url=data_url)

        self.initial_pool_aux_data = auxdata  # This may be required for graphing, etc.

        # Set the start date as well
        self.date_start = auxdata["last_date"]

        all_still_infected = [p for p in initial_situation if p.state == "I"]

        # Last person ID
        self.next_person_id = initial_situation[-1].pers_id + 1

        # Initial conditions: get number of infected per defined area
        # NB! Assumption: we ignore exposed at this point as their number is not known

        # Based on this, we now create a dict (related to a given area) with the person lists
        # The structure of the dict is:
        # { area_id: [pop_size,
        #            [person1, person2, ...]
        #            ]
        # }
        for id in range(8):
            initial_pool[id + 1] = [popmaxsizes[id + 1],
                                    [self.initial_pool_generate_person(p) for p in all_still_infected if p.area_id == id + 1]]

        self.initial_pool = initial_pool
        self.initial_person_last_id = self.next_person_id

    def load_mobility_data(self, location):
        file_path = os.path.relpath(location)
        with open(file_path, "rb") as f:
            mobility_data = pickle.load(f)
        self.mobility_dict = mobility_data

    def do_step(self, logfn=None):

        # Perform checks across all areas
        for k in range(8):
            # Check E -> I
            [p.check_state_e_to_i(self.time, logfn=logfn) for p in self.pool[k + 1][1] if p.state == "E"]

            # Check if will infect
            [p.check_infect_someone(self.time, logfn=logfn) for p in self.pool[k + 1][1] if len(p.will_infect_list) > 0]

            # Check I -> R
            [p.check_state_i_to_r(self.time, logfn=logfn) for p in self.pool[k + 1][1] if p.state == "I"]

        # Increment timer
        self.time += 1
    
    # Number of active cases include exposed AND infected
    def get_number_of_active_cases(self):

        act = []
        for i in range(8):
            act.append(len([p for p in self.pool[i+1][1] if (p.state == "I" or p.state == "E")]))
        return act

    def get_number_of_infected(self):
        inf = []
        for i in range(8):
            inf.append(len([p for p in self.pool[i+1][1] if (p.state == "I")]))
        return inf

    # This is needed to run the simulation more than once
    def get_infected_pool_copy(self):

        pool = copy.deepcopy(self.initial_pool)

        # Once this is done, we need to replace references in every person object to the actual simulation
        # since a copy of the simulation object was created with deepcopy.
        for k in range(8):
            area_id = k + 1
            [setattr(p, "simul", self) for p in pool[area_id][1]]

        return pool

    def do_simulation(self, do_log = True, logfn=print):

        # Reset time
        self.time = 0

        # Set flags
        self.simulation_running = True
        self.simulation_abort = False

        # Simulation log
        self.do_log = do_log

        # Set up the pool with deep copy
        self.pool = self.get_infected_pool_copy()

        # Reset person ID to initial pool person ID
        self.next_person_id = self.initial_person_last_id

        # Keep track of the number of active cases and new cases
        out_act = np.zeros((self.stop_time+1, 9), dtype=int)
        out_new = np.zeros((self.stop_time + 1, 9), dtype=int)
        
        while self.time <= self.stop_time:

            # Check if abort flag gets set
            if self.simulation_abort:
                break

            act_cases = self.get_number_of_active_cases()
            out_act[self.time, :8] = act_cases
            out_act[self.time, 8] = sum(act_cases)

            # This will be used to determine the number of new cases.
            # New cases are defined as follows: increase of infected per day (essentially how many e->i events occured)
            inf_before = self.get_number_of_infected()

            self.do_step(logfn=logfn)

            inf_after = self.get_number_of_infected()
            tot_new_cases = []

            for c in range(8):
                nc = inf_after[c] - inf_before[c]
                tot_new_cases.append(0 if nc < 0 else nc)

            # Because a step was performed at this point, we need to store new cases to previous day statistics
            out_new[self.time-1, :8] = tot_new_cases
            out_new[self.time-1, 8] = sum(tot_new_cases)

        self.simulation_running = False

        if self.simulation_abort:
            return False
        else:
            return (out_act, out_new)

