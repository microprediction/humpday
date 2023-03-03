try:
    import freelunch
    using_freelunch = True
except ImportError:
    using_freelunch = False

if __name__=='__main__':
    print(using_freelunch)