"""
This module defines the agent.
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
