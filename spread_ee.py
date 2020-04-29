from configs.stats import *
from tqdm import tqdm
import pandas as pd
from datetime import timedelta, datetime, date
import numpy as np
import math
import pickle
import time

# Some tunable simulation parameters
EE_SPREAD_SIM_DAYS_BACK = timedelta(days=10)  # Need to run every day to get fresh data
EE_WHEN_INITIAL_POP_INFECT_SOMEONE_FACTOR = 0.8  # Related to generation interval

EE_SIM_DAYS_TO_SIMULATE = 90

# How many simulations?
EE_SIM_NUM_SIMULATIONS = 1000

# This parameter is rather sensitive. It was tuned considering the magnitudes of the probabilities of traveling
# to given regions from given region. Feel free to modify it as needed.
EE_FRACTION_STAYING_ACTIVE = 0.005  # Fraction of those staying active when considering travel probability


# Additional check "will infect" when generating the population based on statistical data
P_ALREADY_INFECTED_INFECT_INITIAL = 0.1

# NB! A modified SEIR model is used for simulation

### Necessary objects for simulation

person_id = 1  # A global variable, each created person is assigned an ID


# At the moment, for "Person" we consider only 3 states: E, I and R
# Assumption: the person will infect someone only after E->I and
# according to probability P(inf) which is P(inf)=R0 for the area
class Person:

    def __init__(self, area_id, t, simul=None):

        global person_id
        self.person_id = person_id
        person_id += 1

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

        # According to probability, assign whether the person will infect someone
        if self.check_if_will_infect():
            self.will_infect = True
            self.will_infect_where = self.get_infection_area()

    # Exposed to infected (time = simulation time)
    def check_state_e_to_i(self, t):

        # We count recovery time from this moment
        if self.state == "E" and self.symptoms_onset == t:

            # The person is now infected and showing symptoms
            self.state = "I"

            # When will the person recover?
            self.recover_time = self.covid19.get_days_to_recovery() + t

        # At this stage we also figure out whether it's time for the person to infect

    # NB! Should account for the situation where R0 > 1.
    # This here is the most crucial step in the whole simulation
    def check_if_will_infect(self):
        return np.random.random() < EE_COVID19_area_R0[self.area_id]

    # This needs an AREAS argument which is the dictionary of lists that contain all the areas with E/I/R persons
    # (from Simulation structure)
    def check_infect_someone(self, t):

        if self.simul is None:
            raise Exception("Simulation object is not associated with this person")

        areadict = self.simul.pool

        if (self.state == "E" or self.state == "I") and self.will_infect and self.will_infect_when > t and \
                areadict[self.will_infect_where][0] >= len(areadict[self.will_infect_where][1]):
            # Only add a new exposed person if area is not saturated (max pop size reached)
            npc = Person(self.will_infect_where, t, self.simul)
            areadict[self.will_infect_where][1].append(npc)

    # Infected to recovered (time = simulation time)
    def check_state_i_to_r(self, t):

        # The person has recovered and is moved out of the active cases pool
        if self.state == "I" and self.recover_time == t:
            self.state = "R"

    def get_infection_area(self):

        # The case when no reference to simulation is available or mobility_dict is missing
        if self.simul is None or self.simul.mobility_dict is None:
            return self.area_id

        # Otherwise, compute the probability of the person infecting another in a different area
        md = self.simul.mobility_dict

        # Stay in own area: a fraction of people remaining in their own area with a chance of interaction
        pop_in_this_area = self.simul.pool[self.area_id][0]
        stay_act = EE_FRACTION_STAYING_ACTIVE * pop_in_this_area

        travel_list = []  # Note that the sequence is important here
        for i in range(8):
            new_area_id = i+1
            if new_area_id != self.area_id:
                travel_prob = np.random.normal(md[self.area_id][new_area_id][0], md[self.area_id][new_area_id][1])
                travel_list.append(travel_prob if travel_prob>0 else 0)  # Add zero for negative values
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
class Covid19SimulationEE:

    def __init__(self, initial_pool):

        self.time = 0  # Timestamp. We need not use dates since we can guarantee that the simulation step is 1 day
        self.date_start = datetime.now()
        self.pool = initial_pool  # This is our initial pool of infected
        self.stop_time = 90  # For how many days to we simulate the spread?

        self.mobility_dict = None  # NB! Need to load this one to get information about mobility between areas

        # We need to check the pool and assign simulation reference to every person
        for i in range(8):
            for p in self.pool[i+1][1]:
                p.simul = self

    def load_mobility_data(self, location):
        with open(location, "rb") as f:
            mobility_data = pickle.load(f)
        self.mobility_dict = mobility_data

    def do_step(self):

        # Perform checks across all areas
        for k in range(8):
            # Check E -> I
            [p.check_state_e_to_i(self.time) for p in self.pool[k + 1][1] if p.state == "E"]

            # Check if will infect
            [p.check_infect_someone(self.time) for p in self.pool[k + 1][1] if p.state == "I"]

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

        # print("Day", self.time, ": number of infected across all areas is", num_infected)
        # print("Day", self.time, ": number of exposed across all areas is", num_exposed)
        
        # print("Day", self.time, ": number of infected across Harju is", num_infected_1)
        # print("Day", self.time, ": number of exposed across Harju is", num_exposed_1)

        # Increment timer
        self.time += 1
    
    # Number of active cases include exposed AND infected
    def get_number_of_active_cases(self):

        act = []
        for i in range(8):
            act.append(len([p for p in self.pool[i+1][1] if (p.state == "I" or p.state == "E")]))
        return act

    def do_simulation(self):
        
        # Keep track of the number of active cases
        out = np.zeros((self.stop_time+1, 9), dtype=int)
        
        while self.time <= self.stop_time:

            act_cases = self.get_number_of_active_cases()
            out[self.time, :8] = act_cases
            out[self.time, 8] = sum(act_cases)

            self.do_step()

        return out


# %% Run this once and save (as instructed below)

# Initial conditions: get number of infected per defined area
# NB! Assumption: we ignore exposed at this point as their number is not known
initial_situation, _ = \
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
    
    # Another important assumption is that we assume that all members of the
    # initial population will infect someone in their own area
    infect_area = p.area_id

    # Need to change some attributes of the person to reflect the above mentioned
    pers.will_infect = False  # By default, the person will not infect anyone (see below for P)
    pers.state = "I"
    pers.recover_time = p.days_to_recovery_from_now
    if pers.check_if_will_infect() and np.random.random() < P_ALREADY_INFECTED_INFECT_INITIAL:
        pers.will_infect = True
        pers.will_infect_where = infect_area
        pers.will_infect_when = math.floor(pers.covid19.get_generation_interval() * EE_WHEN_INITIAL_POP_INFECT_SOMEONE_FACTOR)

        # print("This person will infect someone in", pers.will_infect_when, "days") # Debug

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

print("Running single simulation...")
tim = time.time()

# Create the simulation
Covid19SimEE = Covid19SimulationEE(initial_pool)

# Load mobility data
Covid19SimEE.load_mobility_data("configs/Area_OD_matrix.pkl")

# Stop time
Covid19SimEE.stop_time = EE_SIM_DAYS_TO_SIMULATE

# Run the simulation
out = Covid19SimEE.do_simulation()

# Save data in Excel file
pdata = pd.DataFrame(data=out, columns=["1","2","3","4","5","6","7","8","Total"])
pdata.to_excel("results/test_run_0001.xlsx", sheet_name="exp+inf in areas")

print("Simulation run concluded in", str(time.time()-tim), "seconds. Excel file saved.")

# %% ############### Monte-Carlo Simulations ################
print("Running Monte-Carlo simulations...")
tim = time.time()

outs = []

for k in tqdm(range(EE_SIM_NUM_SIMULATIONS)):

    # Need to reload the list on every simulation
    with open(INIT_DATA_LOC, "rb") as f:
        initial_pool = pickle.load(f)

    # Create the simulation
    Covid19SimEE = Covid19SimulationEE(initial_pool)

    # Load mobility data
    Covid19SimEE.load_mobility_data("configs/Area_OD_matrix.pkl")

    # Stop time
    Covid19SimEE.stop_time = EE_SIM_DAYS_TO_SIMULATE

    # Run the simulation
    out = Covid19SimEE.do_simulation()

    outs.append(out)

# Get the averages and standard deviation
outs_avg = sum(outs)/EE_SIM_NUM_SIMULATIONS
outs_std = np.sqrt(sum(np.power(outs-outs_avg, 2)/(EE_SIM_NUM_SIMULATIONS-1)))

outs_avg_std = np.concatenate([outs_avg, outs_std], axis=1)

# Save data in Excel file
pdata = pd.DataFrame(data=outs_avg_std, columns=["1_mean","2_mean","3_mean","4_mean",
                                            "5_mean","6_mean","7_mean","8_mean",
                                            "Total_mean",
                                            "1_std", "2_std", "3_std", "4_std",
                                            "5_std", "6_std", "7_std", "8_std",
                                            "Total_std"
                                            ])
pdata.to_excel("results/test_run_mc_" + str(EE_SIM_NUM_SIMULATIONS) + ".xlsx", sheet_name="exp+inf in areas")

print("Monte-Carlo run concluded in", str(time.time()-tim), "seconds. Excel file saved.")