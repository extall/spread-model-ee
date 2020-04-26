from configs.stats import *
from tqdm import tqdm
import pandas as pd
import dateparser
from datetime import timedelta, datetime, date
import numpy as np
import enum
import math
import pickle

### Simulation parameters

# Time delta from the present day which we consider the start of simulation
EE_SPREAD_SIM_DAYS_BACK = timedelta(days=10)

# A modified SEIR model is used for simulation

### Necessary objects for simulation

person_id = 1  # A global variable, each created person is assigned an ID


# At the moment, for "Person" we consider only 3 states: E, I and R
# Assumption: the person will infect someone only after E->I and
# according to probability P(inf) which is P(inf)=R0 for the area
class Person:

    def __init__(self, area_id, t):

        global person_id
        self.person_id = person_id
        person_id += 1

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

        # These are properties related to whether the person will actually infect someone else. Only one chance to do it
        # This is determined when the state changes E->I
        self.will_infect = False  # Will the person infect someone else?
        self.will_infect_when = None  # If so, when after E->I conversion will the person do it?
        self.will_infect_where = None  # And where? (which area)

    # Exposed to infected (time = simulation time)
    def check_state_e_to_i(self, t):

        # NB! Apparently only a person with symptoms can infect another.
        # We also count recovery time from this moment
        if self.state == "E" and self.symptoms_onset == t:

            # The person is now infected and showing symptoms
            self.state = "I"

            # When will the person recover?
            self.recover_time = self.covid19.get_days_to_recovery() + t

            # Will the person infect someone?
            if self.check_if_will_infect():
                self.will_infect = True
                self.will_infect_where = self.get_infection_area()

    # NB! Should account for the situation where R0 > 1.
    # This here is the most crucial step in the whole simulation
    def check_if_will_infect(self):
        return np.random.random() < EE_COVID19_area_R0[self.area_id]

    # This needs an AREAS argument which is the dictionary of lists that contain all the areas with E/I/R persons
    # (from Simulation structure)
    def check_infect_someone(self, areadict, t):

        if self.state == "I" and self.will_infect and self.will_infect_when == t and \
                areadict[self.will_infect_where][0] >= len(areadict[self.will_infect_where][1]):
            # Only add a new exposed person if area is not saturated (max pop size reached)
            npc = Person(self.will_infect_where, t)
            areadict[self.will_infect_where][1].append(npc)

    # Infected to recovered (time = simulation time)
    def check_state_i_to_r(self, t):

        # The person has recovered and is moved out of the active cases pool
        if self.state == "I" and self.recover_time == t:
            self.state = "R"

    def get_infection_area(self):
        #print("OD matrix approach not yet implemented, will use own area")
        return self.area_id


# The simulation class. Handles a single simulation
class Covid19SimulationEE:

    def __init__(self, initial_pool):
        self.time = 0  # Timestamp. We need not use dates since we can guarantee that the simulation step is 1 day
        self.date_start = datetime.now()
        self.pool = initial_pool  # This is our initial pool of infected
        self.stop_time = 90  # For how many days to we simulate the spread?

    def do_step(self):

        # Perform checks across all areas
        for k in range(8):
            # Check E -> I
            [p.check_state_e_to_i(self.time) for p in self.pool[k + 1][1] if p.state == "E"]

            # Check if will infect
            [p.check_infect_someone(self.pool, self.time) for p in self.pool[k + 1][1] if p.state == "I"]

            # Check I -> R TODO: Maybe can optimize by constructing the I list once and running two functions over it?
            [p.check_state_i_to_r(self.time) for p in self.pool[k + 1][1] if p.state == "I"]

        # Debug
        num_infected = 0
        num_exposed = 0
        
        for k in range(8):
            num_infected += len([p for p in self.pool[k + 1][1] if p.state == "I"])
            num_exposed += len([p for p in self.pool[k + 1][1] if p.state == "E"])
            
        num_infected_1 = len([p for p in self.pool[1][1] if p.state == "I"])
        num_exposed_1 = len([p for p in self.pool[1][1] if p.state == "E"])

        print("Day", self.time, ": number of infected across all areas is", num_infected)
        print("Day", self.time, ": number of exposed across all areas is", num_exposed)
        
        print("Day", self.time, ": number of infected across Harju is", num_infected_1)
        print("Day", self.time, ": number of exposed across Harju is", num_exposed_1)

        # Increment timer
        self.time += 1

    def do_simulation(self):
        while self.time <= self.stop_time:
            self.do_step()


# %% Run this once and save (as instructed below)

# Initial conditions: get number of infected per defined area
# NB! Assumption: we ignore exposed at this point as their number is not known
initial_situation = \
    ee_parse_infected_dynamically_and_assign_to_areas_until_date(datetime.now() - EE_SPREAD_SIM_DAYS_BACK)

all_still_infected = [p for p in initial_situation if p.state == "I"]

# %% Create dict

# Based on this, we now create a dict (related to a given area) with the person lists
# The structure of the dict is:
# { area_id: [pop_size,
#            [person1, person2, ...]
#            ]
# }

popmaxsizes = ee_covid19_area_population_2016()

# Build the initial pool. After this action, the pool needs to be saved to reduce data processing time.
initial_pool = {}


# Preprocessing function for a person
def convert_person(p):
    # NB! Assumption: we cut generation intervals in half for the initial infected population
    # because there is no data about when the person was exposed and the generation interval is counted from thence
    pers = Person(p.area_id, 0)

    # Need to change some attributes of the person to reflect the above mentioned
    pers.state = "I"
    pers.recover_time = p.days_to_recovery_from_now
    if pers.check_if_will_infect():
        pers.will_infect = True
        pers.will_infect_where = pers.get_infection_area()
        pers.will_infect_when = math.floor(pers.covid19.get_generation_interval() / 2.0)

        print("This person will infect someone in", pers.will_infect_when, "days")

    return pers


# Set up the dictionary
for id in range(8):
    initial_pool[id + 1] = [popmaxsizes[id + 1], [convert_person(p) for p in all_still_infected if p.area_id == id + 1]]

# %% Run this in Spyder, then you can launch this section separately
# (which you should definitely do, simulation set up above is rather costly time wise)
INIT_DATA_LOC = r"C:\Users\Aleksei\Desktop\initial_infected.pkl"

# %% Save the initial dictionary to disk
with open(INIT_DATA_LOC, "wb") as f:
    pickle.dump(initial_pool, f)

# %% ################# LET THE SIMULATION BEGIN #################
with open(INIT_DATA_LOC, "rb") as f:
    initial_pool = pickle.load(f)

# Create the simulation
Covid19SimEE = Covid19SimulationEE(initial_pool)
