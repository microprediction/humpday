
import numpy as np
import matplotlib.pyplot as plt


def mesh2d_(f,*args):
    """ Plot function taking len 2 vector as single argument
          f(xs)
    """
    def g(x,y,*args):
        return f(np.array([x,y]),*args)
    mesh2d(g,*args)


def mesh2d(f,*args):
    """ Plot function taking two arguments
        f(x,y)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y =  np.arange(-0.5, 0.5, 0.005)
    X, Y = np.meshgrid(x, y)
    zs = np.array([ f(x_,y_,*args) for x_,y_ in zip( np.ravel(X), np.ravel(Y)) ])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    
    
 def simplex_surf(g):
    """
         g is a function defined on the 2-simplex 
         We change perspective so as to plot it 
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = [10.0, 8.0]

    # Define the vertices of the triangle in R3
    p3 = np.array([0, 0, 0])
    p1 = np.array([1, 0, 0])
    p2 = np.array([0.5, np.sqrt(3)/2, 0])

    # Define points in the 2-simplex 
    s1s = list()
    s2s = list()
    s3s = list()
    for s1 in np.linspace(0,1.0,50):
      for s2 in np.linspace(0,1.0,50):
          s3 = 1-s1-s2
          if 0 <= s3 <= 1:
              s1s.append(s1)
              s2s.append(s2)
              s3s.append(s3)

    X = np.array([ s1*p1[0] + s2*p2[0] + s3*p3[0] for s1, s2, s3 in zip(s1s,s2s,s3s) ])
    Y = np.array([ s1*p1[1] + s2*p2[1] + s3*p3[1] for s1, s2, s3 in zip(s1s,s2s,s3s) ])
    Z = np.array([ g(u=[s1,s2,s3]) for s1, s2, s3 in zip(s1s,s2s,s3s)])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    from matplotlib import cm
    surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    shadow = ax.plot_trisurf(X, Y, np.zeros_like(Z), cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)

    plt.show()
