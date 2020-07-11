import numpy as np
import math
import ipdb

def compute_yaw_and_pitch(vec):
    norm = np.linalg.norm(vec)
    x, y, z = vec
    # ipdb.set_trace()
    if z<0:
        yaw = math.pi+math.atan2(x,z)
    elif x<0:
        yaw = math.pi*2+math.atan2(x,z)
    else:
        yaw = math.atan2(x,z)
        
    pitch = -math.asin(y/norm)
    return yaw, pitch

# def compute_yaw_and_pitch(vec):
#     x, y, z = vec
#     norm = np.linalg.norm(vec)
#     if z < 0:
#         yaw = math.pi + math.atan(x / z)
#     elif x < 0:
#         yaw = math.pi * 2 + math.atan(x / z)
#     else:
#         yaw = math.atan(x / z)
#     pitch = -math.asin(y / norm)
#     return yaw, pitch