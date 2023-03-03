try:
    import dlib
    using_dlib = True
except ImportError:
    using_dlib = False

if __name__=='__main__':
    print(using_dlib)