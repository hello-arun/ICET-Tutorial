from icet.tools import redlich_kister
import numpy as np
import itertools


def _fit_cs_rk_parameters(cs_rk_parameters):
    """
    Make a polynomial fit of Redlich-Kister parameters on
    the stereographic projection of the 1BZ of FCC.
    """
    A = []
    y = []
    for orientation, parameters in cs_rk_parameters.items():
        projection = _get_projection(orientation)
        A.append(_get_xy_vector(*projection))
        y.append(parameters)

    rk_fit, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return rk_fit


def _get_projection(orientation):
    """
    Stereographic projection of the orientation of an interface.
    Assumes cubic symmetry.
    """
    orientation = np.array(sorted(np.abs(orientation)))
    orientation = orientation / np.linalg.norm(orientation)
    return orientation[:2]


def _get_xy_vector(x, y, deg=2):
    """
    2D polynomial of degree `deg`
    """
    vector = []
    for i, j in itertools.product(range(deg + 1), repeat=2):
        if i + j > deg:
            continue
        vector.append(x**i * y**j)
    return vector


def _k_to_parameter_function(k, cs_fitted_rk_parameters):
    """
    Calculate strain energy at a specific concentration and k point
    using constitutent_strain_functions as fitted with
    _fit_cs_parameters.
    """
    projection = _get_projection(k)
    vector = _get_xy_vector(*projection)
    parameters = np.dot(vector, cs_fitted_rk_parameters)
    return parameters


def custom_k_to_parameter_function(k):
    """
    Create function that precomputes Redlich-Kister parameters
    for a specific k point.
    """
    # Read Redlich-Kister parameters for constituent strain
    cs_data = np.loadtxt('constituent-strain-RK-parameters.data')
    cs_rk_parameters = {}
    for row in cs_data:
        cs_rk_parameters[tuple(int(i) for i in row[:3])] = row[3:]
    cs_fitted_rk_parameters = _fit_cs_rk_parameters(cs_rk_parameters)

    # Define function
    f = _k_to_parameter_function(k, cs_fitted_rk_parameters)
    return f


def custom_strain_energy_function(parameters, c):
    """
    Create function that takes Redlich-Kister parameters and
    concentration and returns corresponding strain energy.
    """
    return redlich_kister(c, *parameters)
