import numpy as np
import math

def compute_yaw_and_pitch(vec):
    norm = np.linalg.norm(vec)
    x, y, z = vec
    
    if y<0:
        yaw = np.pi+np.arctan2(x,y)
    elif x<0:
        yaw = np.pi*2+np.arctan2(x,y)
    else:
        yaw = np.arctan2(x,y)
        
    pitch = -np.arcsin(z/norm)
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