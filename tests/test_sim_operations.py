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
    
# Generator for IDs
def ID_gen():
    id = 0
    while(True):
        yield id
        id += 1

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
    none = 0
    school = 1
    kindergarten = 2
    university = 3
    shopping_center = 4
    residence = 5
    hospital = 6
    club = 7
    bar = 8
    restaurant = 9
    sports_center = 10

# %% NB! Need to figure out the scale
class EE_grid_coordinate:
    unit = (100, "m")  # TODO: This needs to be discussed, cannot really determine without getting some real world data
                       # TODO: With 100m base unit, Tallinn will be divided into about 16000 points with an average of
                       # TODO: 27 people per point.
    def __init__(self, x=0, y=0):
        self.x = 0
        self.y = 0

# TODO: Need a mapper from OSM or Estonia shapefile to EE_grid_coordinates for common objects
# TODO: Also need a class for the grid that takes into account impossible coordinates (will be used for checking
# TODO: actual object/subject coordinates against trespassing onto the sea, for example)

class EE_Person:
    def __init__(self, ID, status="S", loc=EE_locs.none):
        self.ID = ID
        self.status = status  # Susceptible by default
        self.loc = loc
        
    def set_status(self, status="S"):
        self.status = status


class EE_Location:
    def __init__(self, ID, loc=EE_locs.none):
        self.coords = EE_grid_coordinate(0, 0)
        self.ID = ID
        self.type = loc
        
# %% Some params
LARGE_NUMBER = 1500000
NUM_STATUSES = 10
MAT_TYPE = 'uint32'  # Encode states

# %% Play a bit with classes
import random
        
next(timer) # Start timing

# Preprocess locs a bit.
locs = list(EE_locs)

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

'''
IDEA: Can make a simple wrapper class for NUMPY arrays that will
also have a header. Operations will still be fast as the
underlying tables will be manipulated directly with numpy.
'''

# So maybe something like this
# NB! This is partially redoing work already implemented in PANDAS
# but with PANDAS it's hard to say if impl is optimal
class Table(object):
    def __init__(self, header, shp, name="DataTable", dtype='int32'):
        # Header is expected to be a list or tuple of strings
        # having the same horizontal shape as the number of columns
        if type(shp) != tuple or len(shp) != 2 or \
        type(shp[0]) != int or type(shp[1]) != int:
            raise ValueError("The SHP argument must be a tuple with exactly two integers specifying the shape of the table")

        if shp[1] != len(header) or any(not isinstance(item, str) for item in header):
            raise ValueError("The table HEADER list must have the same number of entries as there are columns in the table")

        # Everything OK, let's assign stuff
        self.name = name
        self.shape = shp  # Maybe we can rather inherit from ndarray?
        self.header = list(header)
        self.table = np.zeros(shp, dtype=dtype)
        
    def __getitem__(self, key):
        
        # Check if we're using name instead of index for column
        if type(key) != slice and len(key)>1 and type(key[1]) == str:
                # Named string for second part of key
                return self.table[key[0], self.header.index(key[1])]
                
        return self.table[key]
    
    def __setitem__(self, key, value):
        
        if type(key) != slice and len(key)>1 and type(key[1]) == str:
            self.table[key[0], self.header.index(key[1])] = value
        else:    
            self.table[key] = value
        
    def __repr__(self):
        return " | ".join(self.header) + "\n" + str(self.table)
        
# ###########################################
# %% Without using TABLE
# ###########################################
        
print("NOT USING TABLE")
print("***************")

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

# ###########################################
# %% Using TABLE
# ###########################################
print("USING TABLE")
print("***********")

next(timer) # This starts timing
# Try to create a huge matrix of ints
stat_list = ["index"] + ["status " + str(s) for s in range(NUM_STATUSES-1)]
A = Table(stat_list, (LARGE_NUMBER, NUM_STATUSES), dtype=MAT_TYPE)
print("Creating the Table took", next(timer), "seconds")

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


# ###########################################
# %% Using PANDAS dataframes
# ###########################################

import pandas as pd

print("USING DATAFRAMES")
print("***********")

next(timer) # This starts timing
# Try to create a huge matrix of ints
stat_list = ["index"] + ["status " + str(s) for s in range(NUM_STATUSES-1)]
A = pd.DataFrame(np.zeros((LARGE_NUMBER, NUM_STATUSES)), columns = stat_list)
print("Creating the Table took", next(timer), "seconds")

# Assign IDs as first column
### INDEX NOT NEEDED, IT IS CREATED AUTOMATICALLY

# All people are initially "S"usceptible
next(timer)
A.loc[:, "status 0"] = "S"
print("Assigning S state (actual string) to everyone took", next(timer), "seconds")

# Assign a given property if condition holds
next(timer)
EVERY_NTH_PERSON = 100
c = A.index.values % EVERY_NTH_PERSON == 0  # Every 100th person based on ID
A.loc[c, "status 0"] = "E" # Assign some value (exposed)
print("Assigning a value to every %s person took %f seconds" % (EVERY_NTH_PERSON, next(timer)))

# Assign a given property at throw of dice
next(timer)
c = np.random.random(A.shape[0]) > 0.5  # Probability 50%
A.loc[c, "status 0"] = "I"  # Assign infected state
print("Assigning randomly the ""I"" status to some persons took %f seconds " % next(timer))

# Fetch all "I" persons
next(timer)
P = A[A["status 0"] == "I"]
print("Locating and returning all I persons took %f seconds " % next(timer))
print("Found %d infected persons" % P.shape[0])

# %% CONCLUSION
# Can use Table (essentially, prehashed DataFrame) for fast computations
# But can use DataFrame as a kind of general database that needs to be updated
# not so frequently. In other words, when we deal with DAY simulation, need to
# figure out who can potentially infect whom, and for that we can use Tables since
# they are relatively fast. Then, we can update the statues in DataFrames












