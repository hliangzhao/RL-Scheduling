"""
This module defines the agent.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""


class Agent:
    """
    This class is the father of all kinds of agents, such as ReinforceAgent, FIFOAGent, and DynamicAgent.
    """
    def __init__(self):
        pass

    def get_action(self, obs):
        print('Not implemented')
        exit(1)
