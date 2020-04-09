from configs.stats import *
from tqdm import tqdm
import numpy as np
import enum

# Enumerations


# Structures
class person():

    def __init__(self, loc_id):
        self.time = 0

a = []
for k in tqdm(range(1000000)):
    a.append(ee_stat_gen_age())

with open("test.csv", "w") as f:
    for val in a:
        f.write(str(val) + "\n")
