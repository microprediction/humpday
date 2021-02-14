
M = ['January','February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


def optimizer_name(solver):
    return solver.__name__.replace('_cube','')


def objective_name(objective):
    return objective.__name__.replace('_cube','')