from getjson import getjson


ELO_URL='https://raw.githubusercontent.com/microprediction/optimizer-elo-ratings/main/results/CATEGORY_dDD_nNN.json'


def elo_leaderboard_url(category='classic',n_dim=5, n_trials=130):
    return ELO_URL.replace('CATEGORY',category.lower()).replace('DD',str(n_dim).zfill(2)).replace('NN',str(n_trials))


def get_elo_leaderboard(category='classic',n_dim=5, n_trials=130):
    return getjson(elo_leaderboard_url(category=category,n_dim=n_dim,n_trials=n_trials))


def get_elo_rating(strategy='scipy_powell', category='classic',n_dim=5, n_trials=130):
    s = strategy+'_cube' if '_cube' not in strategy else strategy
    lb = get_elo_leaderboard(category=category,n_dim=n_dim,n_trials=n_trials)
    try:
        ndx = lb['name'].index(s)
        return lb['rating'][ndx]
    except ValueError:
        return None








if __name__=='__main__':
    print(get_elo_rating('pysot_ei_cube'))