import collections

from itertools import product


def create_schedule(param_ranges, verbose=False):
    """
    Args:
        param_ranges: dict of the form {'param1': range(0, 100, 2), 'param2': 1, ...}

    Returns:
        Schedule containing all possible combinations of passed parameter values.
    """

    param_lists = []

    # for each parameter-range pair ('p': range(x)), create a list of the form [('p', 0), ('p', 1), ..., ('p', x)]
    for param, vals in param_ranges.items():
        if isinstance(vals, str):
            vals = [vals]
        # if a single value is passed for param...
        elif not isinstance(vals, collections.Iterable):
            vals = [vals]
        param_lists.append([(param, v) for v in vals])

    # permute the parameter lists
    schedule = [dict(config) for config in product(*param_lists)]

    print('Created schedule containing %d configurations.' % len(schedule))
    if verbose:
        for config in schedule:
            print(config)
        print('-----------------------------------------------')

    return schedule

