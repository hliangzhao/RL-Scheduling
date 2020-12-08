"""
Get the average of per-step-reward.
"""


class AvgPerStepReward:
    """
    TODO: how to understand time?
    """
    def __init__(self, size):
        self.size = size
        self.count = 0
        self.reward_record, self.time_record = [], []
        self.reward_sum, self.time_sum = 0, 0

    def add(self, reward, tm):
        if self.count >= self.size:
            # remove the last if over-sized
            stale_reward = self.reward_record.pop(0)
            stale_time = self.time_record.pop(0)
            self.reward_sum -= stale_reward
            self.time_sum -= stale_time
        else:
            self.count += 1
        self.reward_record.append(reward)
        self.time_record.append(tm)
        self.reward_sum += reward
        self.time_sum += tm

    def add_list_filter_zero(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            if list_time[i] != 0:
                self.add(list_reward[i], list_time[i])
            else:
                assert list_reward[i] == 0

    def get_avg_per_step_reward(self):
        return float(self.reward_sum) / float(self.time_sum)
