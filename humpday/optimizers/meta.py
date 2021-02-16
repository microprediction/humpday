import math
from humpday.comparison.suggestions import recommend

# The equivalent of the "I'm feeling lucky" button on Google search. 

def minimize(objective,n_dim:int,n_trials:int, category='classic', with_count=False):
    """ Minimize an objective, by first choosing a good optimizer then using it.
    :param objective:
    :param n_dim:
    :param n_trials:
    :param category:
    :param with_count:
    :return:
    """
    recommendations = recommend(objective=objective, n_dim=n_dim, n_trials=n_trials, category=category )
    opt = recommendations[0][0]
    return opt(objective, n_dim=n_dim, n_trials=n_trials, with_count=with_count)


if __name__=='__main__':
    import time
    from pprint import pprint

    def my_objective(u):
        time.sleep(0.1)
        return u[0]*math.sin(u[1])

    pprint(minimize(my_objective, n_dim=3, n_trials=130, with_count=True))
