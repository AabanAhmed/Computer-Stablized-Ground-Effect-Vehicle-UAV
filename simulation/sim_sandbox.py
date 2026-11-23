import numpy as np
from scipy.linalg import cont2discrete, solve_discrete_are
import control_module

# State: [h, vz, V, roll, p, pitch, q, yaw, r]
# Input: [taileron_sym, taileron_diff, rudder_sync]
GRID_HEIGHTS = [0.03, 0.08, 0.15, 0.25, 0.38, 0.50]
