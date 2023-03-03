try:
    import nlopt
    using_nlopt = True
except ImportError:
    using_nlopt = False

if __name__=='__main__':
    if not using_nlopt:
        print('pip install nlopt')