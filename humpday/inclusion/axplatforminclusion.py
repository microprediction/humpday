try:
    from ax import optimize
    using_axplatform = True
except ImportError:
    using_axplatform = False

using_axplatform = False
# Until I can shut it up!  https://github.com/facebook/Ax/issues/1486

if __name__=='__main__':
    if not using_axplatform:
        print('pip install ax-platform')