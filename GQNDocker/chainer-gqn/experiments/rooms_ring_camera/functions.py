import numpy as np
import math


def compute_yaw_and_pitch(vec):
    x, y, z = vec
    norm = np.linalg.norm(vec)
    if z < 0:
        yaw = math.pi + math.atan2(x , z)
    elif x < 0:
        yaw = math.pi * 2 + math.atan2(x , z)
    else:
        yaw = math.atan2(x , z)
    pitch = -math.asin(y / norm)

    return yaw, pitch
