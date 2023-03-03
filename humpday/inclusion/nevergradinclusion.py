try:
    import nevergrad as ng
    using_nevergrad = True
except ImportError:
    using_nevergrad = False


if __name__=='__main__':
    if not using_nevergrad:
        print('pip install nevergrad')