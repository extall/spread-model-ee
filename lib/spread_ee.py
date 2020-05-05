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

    def __init__(self, id, area_id, t, simul=None):

        self.person_id = id

        # This is a reference to the simulation. This is needed to fetch some global parameters
        self.simul = simul

        self.covid19 = Covid19()  # Sorry, you have it now
        self.state = "E"  # Default state is exposed
        self.area_id = area_id
        self.timestamp = t  # Simulation time in days (0 = first day, 1 = second, etc.)
        self.symptoms_onset = self.covid19.get_days_to_symptoms() + t

        # NB! Generation interval is counted from moment of infection, so even if the person doesn't infect anyone
        # else, we need to set this parameter upon person generation
        self.generation_interval = self.covid19.get_generation_interval() + t

        # These will be updated in E->I
        self.recover_time = None

        # These are properties related to whether the person will actually infect someone else.
        self.will_infect = False
        self.will_infect_when = self.generation_interval
        self.will_infect_where = None

        # This is reserved for the case of R0 > 1 in which the person will
        # definitely infect at least N people, where N = int(modf(R0)[1]) and
        # with probability modf(R0)[0] one more person
        self.will_infect_more = 0

        # According to probability, assign whether the person will infect someone
        if self.check_if_will_infect():
            self.will_infect = True

            # Only assign if not previously assigned in case r0>=1
            if self.will_infect_where is None:
                self.will_infect_where = self.get_infection_area()

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

        # At this stage we also figure out whether it's time for the person to infect

    # NB! Should account for the situation where R0 > 1.
    # This here is the most crucial step in the whole simulation
    def check_if_will_infect(self):
        # Let us see what happens according to the region's R0
        r0 = self.simul.r0_per_region[self.area_id]
        if r0 > 1:
            self.will_infect_more = int(math.modf(r0)[1])
            self.will_infect_where = self.get_infection_area()  # Need to assign this here for cases r0>=1
            r0 = math.modf(r0)[0]

        return np.random.random() <= r0

    # This needs an AREAS argument which is the dictionary of lists that contain all the areas with E/I/R persons
    # (from Simulation structure)
    def check_infect_someone(self, t, logfn=None):

        if self.simul is None:
            raise Exception("Simulation object is not associated with this person")

        areadict = self.simul.pool

        if (self.will_infect or self.will_infect_more > 0) \
                and t >= self.will_infect_when \
                and areadict[self.will_infect_where][0] >= len(areadict[self.will_infect_where][1]) + self.will_infect_more:

            # Only add a new exposed person if area is not saturated (max pop size reached)
            # Also check mobility restriction probabilities for every case

            # This is the case for r0<=1
            if self.will_infect:

                this_mobility_restricted = False

                # Check mobility to and from Saaremaa
                if self.area_id == 6 or self.will_infect_where == 6:
                    # Check restriction probability
                    if not (np.random.random() <= self.simul.prob_mob_saare_reg_unrestricted):
                        this_mobility_restricted = True
                # Or check inter-region mobility
                elif self.area_id != self.will_infect_where:
                    if not (np.random.random() <= self.simul.prob_mob_regions_unrestricted):
                        this_mobility_restricted = True

                if not this_mobility_restricted:
                    id_to_infect = self.simul.next_person_id
                    npc = Person(id_to_infect, self.will_infect_where, t, self.simul)
                    self.simul.next_person_id += 1  # Need to keep track of this
                    areadict[self.will_infect_where][1].append(npc)
                    if logfn is not None and self.simul.do_log:
                        logfn("Day " + str(t) + ": Person " + str(self.person_id) + " from area " + str(self.area_id) +
                                                " infects a new person " + str(id_to_infect) + " in area " +
                                                str(self.will_infect_where))
                else:
                    if logfn is not None and self.simul.do_log:
                        logfn(
                            "Day " + str(t) + ": Person " + str(self.person_id) + " from area " + str(self.area_id) +
                            " would have infected a new person in area " +
                            str(self.will_infect_where) + ", but movement restrictions prevented that and the person " +
                            "will not infect anyone.")

            # This is the case for "infect more".
            # NB! Assumption: if mobility is restricted, infecting others will fail in all cases
            if self.will_infect_more > 0:

                this_mobility_restricted = False

                # Check mobility to and from Saaremaa
                if self.area_id == 6 or self.will_infect_where == 6:
                    # Check restriction probability
                    if not (np.random.random() <= self.simul.prob_mob_saare_reg_unrestricted):
                        this_mobility_restricted = True
                # Or check inter-region mobility
                elif self.area_id != self.will_infect_where:
                    if not (np.random.random() <= self.simul.prob_mob_regions_unrestricted):
                        this_mobility_restricted = True

                if not this_mobility_restricted:
                    for i in range(self.will_infect_more):
                        id_to_infect = self.simul.next_person_id
                        npc = Person(id_to_infect, self.will_infect_where, t, self.simul)
                        self.simul.next_person_id += 1  # Need to keep track of this
                        areadict[self.will_infect_where][1].append(npc)

                        if logfn is not None and self.simul.do_log:
                            logfn(
                                "Day " + str(t) + ": Person " + str(self.person_id) + " from area " + str(self.area_id) +
                                " infects a new person " + str(id_to_infect) + " in area " +
                                str(self.will_infect_where))
                else:
                    if logfn is not None and self.simul.do_log:
                        logfn(
                            "Day " + str(t) + ": Person " + str(self.person_id) + " from area " + str(self.area_id) +
                            " would have infected " + str(self.will_infect_more) + " person(s) in area " +
                            str(self.will_infect_where) + ", but movement restrictions prevented that and the person " +
                            "will not infect anyone.")

            # There is one chance to infect someone, so we need to nullify the corresponding parameters
            self.will_infect = False
            self.will_infect_more = 0

    # Infected to recovered (time = simulation time)
    def check_state_i_to_r(self, t, logfn=None):

        # The person has recovered and is moved out of the active cases pool
        if self.state == "I" and self.recover_time == t:
            self.state = "R"
            if logfn is not None and self.simul.do_log:
                logfn("Day " + str(t) + ": Person " + str(self.person_id) + " in area " +
                                        str(self.area_id) + " is moved to recovered pool")

    def get_infection_area(self):

        # The case when no reference to simulation is available,
        # mobility_dict is missing, or there is no simulation pool
        if self.simul is None or self.simul.mobility_dict is None or self.simul.pool is None:
            return self.area_id

        # Otherwise, compute the probability of the person infecting another in a different area
        md = self.simul.mobility_dict

        # Stay in own area: a fraction of people remaining in their own area with a chance of interaction
        pop_in_this_area = self.simul.pool[self.area_id][0]
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

        # print("So the person from area", self.area_id, "will infect another in area", chance_area) # Debug

        # Return the most likely infection area
        return chance_area


# The simulation class. Handles a single simulation
class Covid19SimulationEEV1:

    def __init__(self, data_url=None, log_redirect=None, gen_params=None):

        # Simulation timestamp
        self.created_timestamp = datetime.now()

        # Logging enabled by default
        self.do_log = True

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
        pers = Person(self.next_person_id, p.area_id, 0, self)

        # Another important assumption is that we assume that all members of the
        # initial population will infect someone in their own area
        infect_area = p.area_id

        pers.will_infect = False  # By default, the person will not infect anyone (see below for P)
        pers.state = "I"
        pers.recover_time = p.days_to_recovery_from_now
        if pers.check_if_will_infect() and np.random.random() <= self.prob_initial_infect_another:
            pers.will_infect = True
            pers.will_infect_where = infect_area
            pers.will_infect_when = math.floor(pers.covid19.get_generation_interval() * self.init_gen_interval_factor)

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
            [p.check_infect_someone(self.time, logfn=logfn) for p in self.pool[k + 1][1] if (p.will_infect == True or p.will_infect_more > 0)]

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

