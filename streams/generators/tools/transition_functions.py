import math


class Transition:

    @staticmethod
    def sigmoid(cur_point, transition_length):
        p = -(4 / transition_length) * (cur_point - (transition_length / 2))
        y = 1 / (1 + math.pow(math.e, p))
        return y
