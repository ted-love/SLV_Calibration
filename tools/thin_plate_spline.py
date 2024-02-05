import numpy as np
from scipy.spatial.distance import cdist  


class ThinPlateSpline:
    """
    
    This class creates a thin plate spline object.
    Details can be found here:
    https://en.wikipedia.org/wiki/Thin_plate_spline
    
    """
    
    def __init__(self, alpha=0.0):
        """
        
        Parameters
        ----------
        alpha : Float, optional
            Smoothing Paramter, between 0 and 1. The default is 0.0.

        Returns
        -------
        None.

        """
        self._fitted = False                                    # Fitting Flag
        self.alpha = alpha                                      # Smoother
        self.parameters = np.array([], dtype=np.float32)        # Spline constants
        self.control_points = np.array([], dtype=np.float32)    # Points of spline

    def fit(self, X, Y):
        """
        
        Parameters
        ----------
        X : NumPy Array (n_c, d_s)
            Control point at source space (X_c).
        Y : NumPy Array (n_c, d_t)
            Control point in the target space (X_t).

        Returns
        -------
            Self.

        """
        
        X = make_2d(X)
        Y = make_2d(Y)

        n_c, d_s = X.shape
        self.control_points = X
        
        phi = self.radial_distance(X)

        # Creating Matrices
        X_p = np.hstack([np.ones((n_c, 1)), X])

        A = np.vstack([np.hstack([phi + self.alpha * np.identity(n_c),X_p]),
                       np.hstack([X_p.T, np.zeros((d_s + 1, d_s + 1))])])

        Y = np.vstack([Y, np.zeros((d_s + 1, Y.shape[1]))])

        self.parameters = np.linalg.solve(A, Y)
        self._fitted = True

        return self

    def transform(self, X):
        """
        
        Parameters
        ----------
        X : NumPy Array (n, d_s)
            Points in the source space.

        Returns
        -------
        NumPy Array(n, d_t)
            Mapped point in the target space.

        """
      

        X = make_2d(X)

        phi = self.radial_distance(X)  
        
        X = np.hstack([phi, np.ones((X.shape[0], 1)), X])  
        return X @ self.parameters

    def radial_distance(self, X):
        """
        

        Parameters
        ----------
        X : NumPy Array (n, d_s)
            N points in the source space.

        Returns
        -------
        NumPy array (n,n_c)
            The radial distance for each point to a control point, Phi(X).

        """
       
        dist = cdist(X, self.control_points)
        dist[dist == 0] = 1  
        
        return dist**2 * np.log(dist)


def make_2d(array):
    """
    Makes sure array is 2-D, if 1-d, turns into 2-d

    Parameters
    ----------
    array : NumPy array
        DESCRIPTION.

    Returns
    -------
    array : NumPy array

    """

    # Expand last dim in order to interpret this as (n, 1) points
    if array.ndim == 1:
        array = array[:, None]

    return array

