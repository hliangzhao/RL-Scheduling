"""
This module implements the baseline reward, which is used in REINFORCE algorithm.
This module is deep learning framework free.
"""
import numpy as np
import bisect
import itertools


def get_polyfit_bl(all_cum_rewards, all_time, polyfit_order=5):
    """
    Fit all_times (x) and all_cum_rewards (y) with a given order polynomial.
    Return fitted results are baselines.
    """
    assert len(all_cum_rewards) == len(all_time)

    # build one list of all values
    list_cum_rewards = list(itertools.chain.from_iterable(all_cum_rewards))
    list_wall_time = list(itertools.chain.from_iterable(all_time))
    assert len(list_cum_rewards) == len(list_wall_time)

    # normalize the time by the max time
    max_time = float(max(list_wall_time))
    max_time = max(1, max_time)
    list_wall_time = [t / max_time for t in list_wall_time]
    polyfit_model = np.polyfit(list_wall_time, list_cum_rewards, polyfit_order)

    # use n-th order polynomial to get a baseline
    # normalize the time
    max_time = float(max([max(wall_time) for wall_time in all_time]))
    max_time = max(1, max_time)
    baselines = []
    for i in range(len(all_time)):
        normalized_time = [t / max_time for t in all_time[i]]
        baseline = sum(polyfit_model[o] * np.power(normalized_time, polyfit_order - o) for o in range(polyfit_order + 1))
        baselines.append(baseline)

    return baselines


def get_piecewise_linear_fit_bl(all_cum_rewards, all_time):
    """
    Fit all_times (x) and all_cum_rewards (y) with a piecewise linear fit model.
    Return fitted results are baselines.
    """
    assert len(all_cum_rewards) == len(all_time)

    # all unique wall time
    unique_wall_time = np.unique(np.hstack(all_time))

    # for find baseline value for all unique time points
    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0
        for i in range(len(all_time)):
            idx = bisect.bisect_left(all_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                baseline += (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / \
                            (all_time[i][idx] - all_time[i][idx - 1]) * (t - all_time[i][idx]) + \
                            all_cum_rewards[i][idx]

        baseline_values[t] = baseline / float(len(all_time))

    # output n baselines
    baselines = []
    for wall_time in all_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)

    return baselines
