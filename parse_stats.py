# %% Imports
from lib.stats import *
import matplotlib.pyplot as plt
import numpy as np

# %% Parsing (time costly)

initial_situation, datas = \
    ee_parse_infected_dynamically_and_assign_to_areas_until_date(datetime.now())
    
# %% Process the logged data
# Now we create a dict which holds days relative to the first date
# and for every day we will collect information on how many infected there are
# and how many recovered. Thus we will have a curve of active cases according
# to the COVID19 statistics

first_date = datas["first_date"]
the_log = datas["logdata"]

days_active_cases = {}

for row in the_log:
    rel_time = row[1]-first_date
    rel_day = rel_time.days
    diff = 1 if row[3] is "I" else -1
    if rel_day in days_active_cases.keys():
        days_active_cases[rel_day] += diff
    else:
        days_active_cases[rel_day] = diff
        
# Create a numpy array such that will be filled with available data. Will count
# active cases daily. This is needed to plot the final graph.
num_pts = max(days_active_cases.keys()) + 1

t = np.arange(0, num_pts)
y = np.zeros((num_pts,))

# Start filling in the Y (active cases)
actsum = 0
for i in t:
    if int(i) in days_active_cases:
        actsum += days_active_cases[int(i)]
    y[int(i)] = actsum
    
h = plt.figure()
plt.plot(t,y)