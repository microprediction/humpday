
try:
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    using_hebo = True
except ImportError:
    using_hebo = False

if __name__=='__main__':
    if not using_hebo:
        print('pip install hebo')