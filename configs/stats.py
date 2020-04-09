import numpy as np
from numpy.random import choice as ch

# Statistical data: age distribution in Estonia (2019)
EE_stat_age_dist = [(0, 0), (1, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44),
                    (45, 49), (50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 89), (90, 94),
                    (95, 99)]

EE_stat_age_prob = [0.0109, 0.0425, 0.0560, 0.0547, 0.0462, 0.0488, 0.0665, 0.0752, 0.0702, 0.0685, 0.0690, 0.0627,
                    0.0670, 0.0642, 0.0584, 0.0439, 0.0390, 0.0306, 0.0175, 0.0069, 0.0013]


# Randomly draw an age from the probability distribution
def ee_stat_gen_age():
    ind = ch(np.arange(len(EE_stat_age_dist)), p=EE_stat_age_prob)
    return ch(np.arange(EE_stat_age_dist[ind][0], EE_stat_age_dist[ind][1]+1))
