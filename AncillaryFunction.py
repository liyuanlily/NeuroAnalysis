# Ancillary functions
import numpy as np


def normalize(x):
    """
    Normalize a vector x

    Input:
    x: numpy.ndarry shape: (n, )
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def first_index(x, y, condition='>', order='forward'):
    """
    Get the first index that meets the condition

    Inputs:
    x: numpy.ndarry shape: (n, )
    y: numpy.ndarry shape: (n, )
    condition: should be one of '>','>=','<','<=','=='
    order: choose from 'forward' and 'backward', the order we look through the array
    """

    if order == 'forward':
        for i in range(len(x)):
            if condition == '>':
                if x[i] > y[i]:
                    return i
            elif condition == '>=':
                if x[i] >= y[i]:
                    return i
            elif condition == '<':
                if x[i] < y[i]:
                    return i
            elif condition == '<=':
                if x[i] <= y[i]:
                    return i
            else:
                if x[i] == y[i]:
                    return i
        return len(x) - 1
    else:
        for i in range(len(x) - 1, -1, -1):
            if condition == '>':
                if x[i] > y[i]:
                    return i
            elif condition == '>=':
                if x[i] >= y[i]:
                    return i
            elif condition == '<':
                if x[i] < y[i]:
                    return i
            elif condition == '<=':
                if x[i] <= y[i]:
                    return i
            else:
                if x[i] == y[i]:
                    return i
        return 0
