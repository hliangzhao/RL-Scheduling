import pprint


def get_executor_interval_map():
    """
    What is this?
    args.executor_data_point example: [5, 10, 20, 40, 50, 60, 80, 100]
    args.exec_cap example: 100
    :return:
    """
    executor2interval = dict()

    # the left most
    # i = 0 ---> e: 0 ~ args.executor_data_point[0]
    for e in range(executor_data_point[0] + 1):
        executor2interval[e] = (executor_data_point[0], executor_data_point[0])

    # the center (without head and tail)
    # i: 0 ~ len(args.executor_data_point) - 2 --->
    for i in range(len(executor_data_point) - 1):
        # e: executor_data_point[i] + 1 ~ executor_data_point[i + 1] - 1
        for e in range(executor_data_point[i] + 1, executor_data_point[i + 1]):
            executor2interval[e] = (executor_data_point[i], executor_data_point[i + 1])
        # e: executor_data_point[i + 1]
        e = executor_data_point[i + 1]
        executor2interval[e] = (executor_data_point[i + 1], executor_data_point[i + 1])

    # the residual
    if exec_cap > executor_data_point[-1]:
        # e: executor_data_point[i_max] + 1 ~ args.exec_cap
        for e in range(executor_data_point[-1] + 1, exec_cap + 1):
            executor2interval[e] = (executor_data_point[-1], executor_data_point[-1])

    return executor2interval


executor_data_point = [5, 10, 20, 40, 50, 60, 80, 100]
exec_cap = 100

if __name__ == '__main__':
    pprint.pprint(get_executor_interval_map())
