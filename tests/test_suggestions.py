from humpday import suggest


def test_suggest():
    for n_dim in range(3,7):
        for n_trials in [22,117,444]:
            for n_seconds in [0.01,10,100]:
                sg = suggest(n_dim=n_dim, n_trials=n_trials, n_seconds=n_seconds, category = 'classic')
                assert len(sg)>=1


if __name__=='__main__':
   test_suggest()