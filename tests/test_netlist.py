import random

import pytest
import numpy as np
import circuitlib as clb

from scipy import sparse
from circuitlib.element import Resistor

# For our initial tests we will consider a (R||R) + R circuit
def test_matrix_w_no_kvl():
    netlist = clb.Netlist()
    r_val = 100
    netlist.R([1,2], r_val, None, None)
    netlist.R([1,2], r_val, None, None)
    netlist.R([2,0], r_val, None, None)
    matrix = sparse.coo_matrix([
        [2/r_val, -2/r_val],
        [-2/r_val, 3/r_val]
    ])
    dense_mat = matrix.todense()
    dense_netlist_mat = netlist.matrix()[0].todense()

    assert np.all(dense_mat == dense_netlist_mat)

@pytest.mark.skip(reason='need to fix when API is stable')
def test_matrix_w_one_kvl():
    netlist = clb.Netlist()
    r_val = 100
    netlist.R([1,2], r_val, None, False)
    netlist.R([1,2], r_val, None, False)
    netlist.R([2,0], r_val, None, True)
    matrix = sparse.coo_matrix([
        [2/r_val, -2/r_val, 0],
        [-2/r_val, 2/r_val, 1],
        [0,     1,      r_val]
    ])
    dense_mat = matrix.todense()
    dense_netlist_mat = netlist.matrix()[0].todense()

    assert np.all(dense_mat == dense_netlist_mat)

def test_lhs_matrix_w_voltage_source():
    netlist = clb.Netlist()
    netlist.V([1,0], 5)

    matrix = sparse.coo_matrix([
        [0, 1],
        [1, 0]
    ])
    dense_mat = matrix.todense()
    dense_netlist_mat = netlist.matrix()[0].todense()

    assert np.all(dense_mat == dense_netlist_mat)

@pytest.mark.skip(reason='need to fix when API is stable')
def test_rhs_matrix_w_voltage_source():
    netlist = clb.Netlist()
    netlist.V([1,0], 5)

    matrix = sparse.coo_matrix([
        [0],[5],
    ])
    dense_mat = matrix.todense()
    dense_netlist_mat = netlist.matrix()[1].todense()

    assert np.all(dense_mat == dense_netlist_mat)

def test_lhs_matrix_w_voltage_source_resistors():
    netlist = clb.Netlist()
    r_val = 100
    netlist.V([1,0], 5)
    netlist.R([1,2], r_val, None, None)
    netlist.R([1,2], r_val, None, None)
    netlist.R([2,0], r_val, None, None)

    matrix = sparse.coo_matrix([
        [2/r_val, -2/r_val, 1],
        [-2/r_val, 3/r_val, 0],
        [1,        0,       0]
    ])
    dense_mat = matrix.todense()
    dense_netlist_mat = netlist.matrix()[0].todense()

    assert np.all(dense_mat == dense_netlist_mat)

@pytest.mark.skip(reason='need to fix when API is stable')
def test_lhs_matrix_w_voltage_source_resistors_one_kvl():

    r_val = 100

    netlist = clb.Netlist()
    netlist.V([1,0], 5)
    netlist.R([1,2], r_val, None, False)
    netlist.R([1,2], r_val, None, False)
    netlist.R([2,0], r_val, None, True)

    matrix = sparse.coo_matrix([
        [2/r_val, -2/r_val, 1,     0],
        [-2/r_val, 2/r_val, 0,     1],
        [1,              0, 0,     0],
        [0,              1, 0, r_val]
    ])
    dense_mat = matrix.todense()
    dense_netlist_mat = netlist.matrix()[0].todense()

    assert np.all(dense_mat == dense_netlist_mat)

@pytest.mark.skip(reason='need to fix when API is stable')
def test_rhs_matrix_w_voltage_source_resistors_one_kvl():

    r_val = 100

    netlist = clb.Netlist()
    netlist.V([1,0], 5)
    netlist.R([1,2], r_val, None, False)
    netlist.R([1,2], r_val, None, False)
    netlist.R([2,0], r_val, None, True)

    matrix = sparse.coo_matrix([
        [0],
        [0],
        [5],
        [0]
    ])
    dense_mat = matrix.todense()
    dense_netlist_mat = netlist.matrix()[1].todense()

    assert np.all(dense_mat == dense_netlist_mat)
