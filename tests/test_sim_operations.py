# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:37:31 2020

@author: Aleksei
"""

# Timing
import time

def time_me():
    start_time = time.time()
    while True:
        last_time = time.time()
        yield time.time() - start_time
        start_time = last_time

""" IDEA
# We can create the population with classes, e.g., have classes for person
# and location. But for optimizing lookup etc. we can convert the object
# lists to NUMPY arrays using hashing (IDs representing traits)
#
# In other words, we will be creating database tables for quick lookup,
# because NUMPY arrays are efficient.
#
# TODO: are pandas dataframes as efficient? If yes, can improve semantics
# of data tables a lot.
"""

""" FURTHER FINDINGS
While looking for PANDAS vs NUMPY, found an interesting paper
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.537.723
through the repo here: https://github.com/UDST/synthpop

The code allows to generate a synthetic population and can be helpful to us
"""

# %% Timer initialization (it is a generator)
timer = time_me()

# %% Classes
import enum
class EE_locs(enum.Enum):
    none = "none"
    school = "school"
    shopping_center = "shopping center"
    university = "university"
    residence = "residence"
    club = "club"
    sports_center = "sports center"
    

class EE_Person:
    def __init__(self, ID, status="S", loc=EE_locs.none):
        self.ID = ID
        self.status = status  # Susceptible by default
        self.loc = loc
        
    def set_status(self, status="S"):
        self.status = status

class EE_Location:
    def __init__(self, ID, loc=EE_locs.none):
        self.coords = (0, 0)
        self.ID = ID
        self.type = loc
        
# %% Some params
LARGE_NUMBER = 1500000
NUM_STATUSES = 10
MAT_TYPE = 'uint32'  # Encode states

# %% Play a bit with classes
import random

# Use a generator for ids
def ID_gen():
    id = 0
    while(True):
        yield id
        id += 1
        
next(timer) # Start timing

# Preprocess locs a bit. TODO: should be easier to do this!
locs = [l[1] for l in list(enumerate(EE_locs))]

# Person ID
pid = ID_gen()

# Some random location IDs for now
loc_ids = range(100)

population = [EE_Person(i, "S", EE_Location(random.choice(loc_ids), random.choice(locs))) for i in range(LARGE_NUMBER)]

print("Generating population list with objects took %f seconds" % next(timer))

# List comprehension: assign randomly "I" to the population
next(timer)
[p.set_status(random.choice(["S", "I"])) for p in population if p.status == "S"]
print("Setting randomly the ""I"" status took %f seconds" % next(timer))

# Getting all people who are infected in school
next(timer)
inf_in_school = [p for p in population if p.status == "I" and p.loc.type == EE_locs.school]
print("Querying for the ""I"" status in schools took %f seconds" % next(timer))


# %% Play with matrices
import numpy as np

next(timer) # This starts timing
# Try to create a huge matrix of ints
A = np.zeros((LARGE_NUMBER, NUM_STATUSES), dtype=MAT_TYPE)
print("Creating the matrix took", next(timer), "seconds")

# Assign IDs as first column
next(timer)
A[:,0] = np.arange(0, LARGE_NUMBER)
print("Creating the range took", next(timer), "seconds")

# All people are initially "S"usceptible
next(timer)
A[:, 1] = ord("S")
print("Assigning S state to everyone took", next(timer), "seconds")

# Assign a given property if condition holds
next(timer)
EVERY_NTH_PERSON = 100
c = A[:,0] % EVERY_NTH_PERSON == 0  # Every 100th person based on ID
A[c, 1] = ord("E") # Assign some value (exposed)
print("Assigning a value to every %s person took %f seconds" % (EVERY_NTH_PERSON, next(timer)))

# Assign a given property at throw of dice
import random
next(timer)
c = np.random.random(A.shape[0]) > 0.5  # Probability 50%
A[c, 1] = ord("I")  # Assign infected state
print("Assigning randomly the ""I"" status to some persons took %f seconds " % next(timer))

# Fetch all "I" persons
next(timer)
c = A[:, 1] == ord("I")
P = A[c,:]
print("Locating and returning all I persons took %f seconds " % next(timer))
print("Found %d infected persons" % P.shape[0])

