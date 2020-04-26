import numpy as np
from scipy.stats import weibull_min
from numpy.random import choice as ch
import pandas as pd
import dateparser
from datetime import timedelta, datetime, date
import math
import sys

# Some URLs where to source data from
EE_URL_COVID19_TEST_RESULTS_PER_COUNTY = "https://opendata.digilugu.ee/opendata_covid19_test_results.csv"

# This may be needed for some particular applications, since comparing INTs is much cheaper than comparing strings
EE_counties_EHAK = {"Harju maakond": 37,
                    "Hiiu maakond": 39,
                    "Ida-Viru maakond": 45,
                    "Jõgeva maakond": 50,
                    "Järva maakond": 52,
                    "Lääne maakond": 56,
                    "Lääne-Viru maakond": 60,
                    "Põlva maakond": 64,
                    "Pärnu maakond": 68,
                    "Rapla maakond": 71,
                    "Saare maakond": 74,
                    "Tartu maakond": 79,
                    "Valga maakond": 81,
                    "Viljandi maakond": 84,
                    "Võru maakond": 87}

# Latest population number per county stats available (2016 year end)
EE_counties_EHAK_pop_num_2016_year_end = {37: 582556,
                                          39: 9335,
                                          45: 143880,
                                          50: 30840,
                                          52: 30378,
                                          56: 24301,
                                          60: 58856,
                                          64: 27963,
                                          68: 82535,
                                          71: 34085,
                                          74: 33307,
                                          79: 145550,
                                          81: 30084,
                                          84: 47288,
                                          87: 33505}

# Per agreement, these are the areas that we consider for analyzing specifically the spread of COVID19
EE_COVID19_spread_areas = {1: (37,),  # Harjumaa
                           2: (56, 71, 52, 60),  # Lääne-, Rapla-, Järva- ja Lääne-Virumaa
                           3: (45,),  # Ida-virumaa
                           4: (79,),  # Tartumaa
                           5: (68,),  # Pärnumaa
                           6: (39, 74),  # Saare- ja Hiiumaa
                           7: (50, 84),  # Viljandi- ja Jõgevamaa
                           8: (64, 81, 87)  # Põlva-, Valga- ja Võrumaa
                           }

# Area R0
EE_COVID19_area_R0 = {1: 1.0,
                      2: 0.8,
                      3: 0.8,
                      4: 0.8,
                      5: 0.8,
                      6: 0.6,
                      7: 0.8,
                      8: 0.8
                      }


# We need population figures for each area
def ee_covid19_area_population_2016():
    pops = {}
    for k in range(8):
        popsize = 0
        for i in EE_COVID19_spread_areas[k + 1]:
            popsize += EE_counties_EHAK_pop_num_2016_year_end[i]
        pops[k + 1] = popsize
    return pops

# Reverse lookup for county vs area. Speeds up things a bit.
def ee_covid19_county_area_reverse_lookup():
    revl = {}
    for key, val in EE_COVID19_spread_areas.items():
        for ehak in val:
            revl[ehak] = key
    return revl


# Get county EHAK from county name. Format: "Harju maakond", "Saare maakond", etc
def ee_get_county_ehak(name):
    ehak = -1  # Check against this value to determine whether the name lookup failed
    if name in EE_counties_EHAK:
        ehak = EE_counties_EHAK[name]
    return ehak


# Statistical data: age distribution in Estonia (2019)
EE_stat_age_dist = [(0, 0), (1, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44),
                    (45, 49), (50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 89), (90, 94),
                    (95, 99)]

EE_stat_age_prob = [0.0109, 0.0425, 0.0560, 0.0547, 0.0462, 0.0488, 0.0665, 0.0752, 0.0702, 0.0685, 0.0690, 0.0627,
                    0.0670, 0.0642, 0.0584, 0.0439, 0.0390, 0.0306, 0.0175, 0.0069, 0.0013]


# Randomly draw an age from the probability distribution (distribution configured automatically using weights in Python)
def ee_stat_gen_age():
    ind = ch(np.arange(len(EE_stat_age_dist)), p=EE_stat_age_prob)
    return ch(np.arange(EE_stat_age_dist[ind][0], EE_stat_age_dist[ind][1] + 1))


# COVID19 related information in Estonia: functions
def ee_read_covid19_stats():
    # Get the data from the internet
    data = pd.read_csv(EE_URL_COVID19_TEST_RESULTS_PER_COUNTY)

    # Sort the data by ResultTime: we will use this later to determine the alleged number of people that recovered
    data = data.sort_values(by=["ResultTime"])

    return data


# COVID19 status per area at the beginning of simulation based on test results according to official Estonian stats
# Here, we need to be a bit careful. The following assumptions are in effect:
# 1. We assume that each person takes the test only once.
# 2. We discard any test that references an unknown county.
# 3. We assume that only individuals with symptoms take the test. This means we only count "infected" persons that will
#    be converted to proper person class later in the simulation.
# 4. We draw statistics for "time to recovery" and apply them to each individual tested positive
#    so that we can get a more or less accurate starting point for the simulation where we set up the initial amount
#    of infected per area ID=1...8 (see above). Thus, if a person recovers before the simulation is started, we do not
#    consider them in the simulation (maybe count towards recovered?)
# 5. We consider the worst case scenario by drawing from "symptoms to recovery" distribution since we do not take into
#    account how many days the person has already spent with having symptoms
class InfectedPerson:

    # Need two pieces of information: when did the test return
    def __init__(self, pers_id, when_test, area_id):
        self.pers_id = pers_id  # Some ID, should be unique. You can use a counter.
        self.state = "I"  # Infected, but can turn to recovered with passing of time
        self.area_id = area_id  # Area ID, look above to see how this is related to counties
        self.covid19 = Covid19()  # A personal virus
        self.timestamp = when_test  # datetime when tested positive

        dtr = self.covid19.get_days_to_recovery()
        self.days_to_recovery = dtr  # We do not keep stats about recovered people (why?)
        self.days_to_recovery_from_now = dtr  # At first, it's the same, but we'll update it for later use

        # Debug
        print("Person ", self.pers_id, " was diagnosed with COVID19 on ", self.timestamp, " in the area ", self.area_id)

    # new_date must be relative datetime
    def check_recovered(self, new_date):
        days_passed = new_date - self.timestamp
        self.days_to_recovery_from_now = self.days_to_recovery - days_passed.days
        if days_passed.days > self.days_to_recovery:
            self.state = "R"  # Congratulations. You have (probably) survived
            print("Person ", self.pers_id, " recovered from COVID19 on ", new_date, " in the area ", self.area_id)


# Once the person class is set up, we proceed with parsing the data. In the end, we have a list of people that are
# either currently infected, or have recovered. In the next stage, one needs to filter out recovered people and
# create separate lists for every area. We parse the list until the indicated date is reached.
def ee_parse_infected_dynamically_and_assign_to_areas_until_date(mydate, do_log=False, log_loc=""):
    # Parse the dataframe
    print("Parsing the data about COVID19 occurence in Estonia according to official information...")

    # Logging: redirect print to file. TODO: This can lead to problems. Avoid logging using this method in general
    # The main problem is that if the user cancels the script execution, stdout is not returned to its previous state
    if do_log:
        def_stdout = sys.stdout
        if log_loc == "":
            log_loc = "parse_infected_test_stats.log"
        sys.stdout = open(log_loc, "w")

    # First, get the fresh stats
    data = ee_read_covid19_stats()

    # Infected people
    the_infected = []

    # Infected counter/ID
    the_inf_id = 1

    # Generate the reverse lookup for the areas
    county_to_area = ee_covid19_county_area_reverse_lookup()

    for row in data.iterrows():

        # Here we do a few checks. First, which of the people in the infected list should recover on this day
        current_day = dateparser.parse(row[1]["ResultTime"])

        [person.check_recovered(current_day) for person in the_infected if person.state == "I"]

        # Stop the loop if we have reached the desired date
        day_diff = mydate - current_day
        if day_diff.days <= 0:
            break

        # Check county and also we are only interested in positive tests
        county = ee_get_county_ehak(row[1]["County"])
        if county is not -1 and row[1]["ResultValue"] == "P":
            # We have found an infected person. Let us create the person and add it to the list
            person = InfectedPerson(the_inf_id, current_day, county_to_area[county])
            the_infected.append(person)
            the_inf_id += 1

    if do_log:
        sys.stdout.close()
        sys.stdout = def_stdout

    print("Finished parsing.")

    return the_infected


# Virus related statistics
class Virus:

    def __init__(self):
        self.about = "A general enough class that provides the template to model other viruses"

        self.days_to_recovery_params = None
        self.generation_interval_params = None
        self.days_to_symptoms_params = None

    def get_days_to_recovery(self):
        if self.days_to_recovery_params is not None:
            return self.days_to_recovery_params[0](self.days_to_recovery_params[1])
        else:
            raise Exception("Days to recovery parameter is undefined for this virus.")

    def get_generation_interval(self):
        if self.generation_interval_params is not None:
            return self.generation_interval_params[0](self.generation_interval_params[1])
        else:
            raise Exception("Generation interval parameter is undefined for this virus.")

    def get_days_to_symptoms(self):
        if self.days_to_symptoms_params is not None:
            return self.days_to_symptoms_params[0](self.days_to_symptoms_params[1])
        else:
            raise Exception("Days to recovery parameter is undefined for this virus.")

    # Distribution helpers: independent of underlying implementation
    # Assumptions: whenever day information is returned, it is always rounded to *next* integer (ceil)

    # Log-normal
    @staticmethod
    def lognormal(params):
        # params[0] = mean, params[1] = sigma
        return math.ceil(np.random.lognormal(params[0], params[1]))

    # Weibull
    @staticmethod
    def weibull(params):
        # params[0] = shape, params[1] = scale
        return math.ceil(weibull_min.rvs(params[0], loc=0, scale=params[1], size=1))


# COVID19 virus statistics
class Covid19(Virus):

    def __init__(self):
        super().__init__()
        self.about = "A class that provides COVID-19 related statistics"

        # You can technically change this externally; the format of the argument is
        # (object.distribution_helper, (param1, param2, ...)) where paramN is the Nth parameter of the distribution
        self.days_to_recovery_params = (self.lognormal, (2.6, 0.4))
        self.days_to_symptoms_params = (self.lognormal, (1.6, 0.42))  # I.e., incubation phase
        self.generation_interval_params = (self.weibull, (2.8, 5.8))
