"""
This script checks the column correlation for a set of clustervectors,
it will be asserted that no columns are not highly correlated
"""

from icet import ClusterSpace
from ase.build import bulk
import numpy as np
import random


def generateRandomStructure(structure_prim, subelements):
    """
    Generate a random structure with structure_prim as a base
    and fill it randomly with elements in subelements
    """

    structure = structure_prim.copy().repeat(8)

    for at in structure:
        element = random.choice(subelements)
        at.symbol = element

    return structure


def generateCVSet(n, structure_prim, subelements, clusterspace):
    """
    generate a set of clustervectors from clusterspace
    """
    clustervectors = []
    for i in range(n):
        conf = generateRandomStructure(structure_prim, subelements)
        cv = clusterspace.get_cluster_vector(conf)
        clustervectors.append(cv)

    return clustervectors


def getColumnCorrelation(i, j, cv_matrix):
    """
    Returns the correlation between column i and j

    cv_matrix: numpy matrix
    """
    col_i = cv_matrix[:, i]
    col_j = cv_matrix[:, j]

    corr = np.dot(col_i, col_j) / \
        (np.linalg.norm(col_i) * np.linalg.norm(col_j))

    return corr


def checkNoCorrelation(cvs, tol=0.99):
    """
    check that no column in cvs are above tolerance
    """
    cvs_matrix = np.array(cvs)
    no_correlated_columns = True
    for i in range(len(cvs[0])):
        if i == 0:  # dont loop over zerolet since this is always ones
            continue
        for j in range(len(cvs[0])):
            if j <= i:
                continue
            corr = getColumnCorrelation(i, j, cvs_matrix)
            if corr > tol:
                print("columns {} and {} were correletated with {}".format(
                    i, j, corr))
                no_correlated_columns = False
    assert no_correlated_columns


def test_no_column_correlation_for_ternary():

    subelements = ['H', 'He', 'Pb']
    structure_row = bulk('H', 'fcc', a=1.01)
    cutoffs = [2.8] * 1

    clusterspace = ClusterSpace(structure_row, cutoffs, subelements)
    cvs = generateCVSet(40, structure_row, subelements, clusterspace)
    checkNoCorrelation(cvs)
