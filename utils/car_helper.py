####
# This contains some helper methods to use in Autonomous RC Car Project
###
import math


# http://www.iquilezles.org/apps/graphtoy/
# round((log(abs(x)+1))/(log(2))*(x)/(abs(x)))

def convert_steering_angle_to_buckets(steer_angle):
    # Convert steering angle to buckets to feed into RNN
    # The reason is the data set is unbalanced (biased toward 0* steer angle)
    # http://cs.stackexchange.com/questions/33493/balanced-weight-distribution-in-bins-buckets

    # Borrowed from Car-puter Project from Otavio Good
    # https://github.com/otaviogood/carputer/
    return round(math.copysign(math.log(abs(steer_angle) + 1.25, 1.25), steer_angle) + 5)


def convert_buckets_to_steer_angle(a):
    # Reverse the function that buckets the steering for neural net output.
    # This is half in filemash.py and a bit in convnet02.py (maybe I should fix)
    # steers[-1] -= 90
    # log_steer = math.copysign( math.log(abs(steers[-1])+1, 2.0) , steers[-1])
    #  0  -> 0, 1  -> 1, -1 -> -1, 2  -> 1.58, -2 -> -1.58, 3  -> 2
    # gtVal = gtArray[i] + 7
    steer = a - 7
    original = steer
    steer = abs(steer)
    steer = math.pow(1.25, steer)
    steer -= 1.25
    steer = math.copysign(steer, original)
    # steer += 90.0
    return steer


