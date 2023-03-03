
try:
    from pybobyqa import solve
    using_bobyqa = True
except ImportError:
    using_bobyqa = False


if __name__=='__main__':
    if not using_bobyqa:
        print('pip install Py-BOBYQA')