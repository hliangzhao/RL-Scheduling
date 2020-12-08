"""
This module defines the tensorflow logger.
"""
import tensorflow as tf
from time import gmtime, strftime
from params import args


class Logger:
    """
    The tensorflow logger.
    """
    def __init__(self, sess, var_list):
        """
        :param sess: need-to-save session
        :param var_list: set as
                ['total_loss', 'entropy', 'value_loss', 'episode_length', 'avg_reward_per_sec',
                'sum_reward', 'reset_prob', 'num_jobs', 'reset_hit', 'avg_job_duration', 'entropy_weight']
        """
        self.sess = sess
        self.summary_vars = []
        for var_name in var_list:
            tf_var = tf.Variable(0.)
            tf.summary.scalar(var_name, tf_var)
            self.summary_vars.append(tf_var)
        self.summary_ops = tf.summary.merge_all()   # save all vars in self.summary_vars into disk
        # use the file path to save session graph
        self.writer = tf.summary.FileWriter(args.result_folder + 'logs/' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    def log(self, epoch_idx, values):
        """
        Save the values of summ vars of this epoch into disk incrementally.
        """
        assert len(self.summary_vars) == len(values)
        summary_str = self.sess.run(
            self.summary_ops,
            feed_dict={self.summary_vars[i]: values[i] for i in range(len(values))}
        )
        self.writer.add_summary(summary=summary_str, global_step=epoch_idx)
        self.writer.flush()
