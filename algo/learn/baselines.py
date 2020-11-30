"""
This module implements the baseline reward, which is used in REINFORCE algorithm.
This module is deep learning framework free.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
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
    list_cum_rewards = list(itertools.chain.from_iterable(all_cum_rewards))
    list_time = list(itertools.chain.from_iterable(all_time))

    # normalize the time
    list_time = [t / max(1, float(max(list_time))) for t in list_time]
    polyfit_model = np.polyfit(list_time, list_cum_rewards, polyfit_order)

    # get baseline of each time
    max_time = max(1, float(max([max(time) for time in all_time])))
    baselines = []
    for i in range(len(all_time)):
        normalized_time = [t / max_time for t in all_time[i]]
        baselines.append(
            sum(polyfit_model[o] * np.power(normalized_time, polyfit_order - o) for o in range(polyfit_order + 1))
        )
    return baselines


def get_piecewise_linear_fit_bl(all_cum_rewards, all_time):
    """
    Fit all_times (x) and all_cum_rewards (y) with a piecewise linear fit model.
    Return fitted results are baselines.
    """
    assert len(all_cum_rewards) == len(all_time)
    unique_time = np.unique(np.hstack(all_time))
    # find baseline values for all unique time points
    baseline_values = {}
    for t in unique_time:
        baseline = 0
        for i in range(len(all_time)):
            idx = bisect.bisect_left(all_time[i], t)
            if idx == 0 or all_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            else:
                baseline += (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / (all_time[i][idx] - all_time[i][idx - 1]) * \
                            (t - all_time[i][idx]) + all_cum_rewards[i][idx]
        baseline_values[t] = baseline / float(len(all_time))
    baselines = []
    for time in all_time:
        baselines.append(
            np.array([baseline_values[t] for t in time])
        )
    return baselines
