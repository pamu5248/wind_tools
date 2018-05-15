"""geometry module
"""
import numpy as np

def wrap_180(x):
    """
    Wrap an angle to between -180 and 180
    """  
    
    x = np.where(x<=-180.,x+360.,x)
    x = np.where(x>180.,x-360.,x)

    return(x)

def wrap_360(x):
    """
    Wrap an angle to between 0 and 360
    """  
    
    x = np.where(x<0.,x+360.,x)
    x = np.where(x>=360.,x-360.,x)

    return(x)

def pat_degrees_between_two_instruments(instrument1_direction,instrument2_direction):
# 20180427 1:15pm
    degrees1 = instrument1_direction.values
    degrees2 = instrument2_direction.values
    angle_between = degrees1-degrees2
    angle_between = np.array(angle_between)
    
    for index in range(len(angle_between)):
        if (angle_between[index] > 180) and (angle_between[index] <= 360):
            angle_between[index] = 360-angle_between[index]
        elif (angle_between[index] < 180) and (angle_between[index] >= 0):
            angle_between[index] = -1*angle_between[index]
        elif (angle_between[index] < 0) and (angle_between[index] >= -180):
            angle_between[index] = -1*angle_between[index]
        elif (angle_between[index] < -180) and (angle_between[index] >= -360):
            angle_between[index] = -1*(360+angle_between[index])
    return angle_between