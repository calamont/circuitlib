"""
There are two aspects of the library we need to test. Firstly, we
want to ensure that the API the user interacts with works as expected.
Secondly, we need to check that the MNA is correctly simulating the circuits
specified by the user.

We can kill two birds with one stone by specifying the circuit through the
API, then evaulating the output against an analytical solution for the circuit.

It might be cleanest to test a lot of this using a single relatively complex
circuit with a known transfer function, such as a bandpass filter. A RLC bandbass
filter would allow us to test all components, as well as solutions to the node
voltages and currents. We could have a series configuration to test the node
voltages, and a parallel configuration to test the node currents.
"""

import pytest

import numpy as np
import circuitlib as clb

@pytest.fixture()
def voltage():
    return 5

@pytest.fixture()
def current():
    return 10e-3

@pytest.fixture()
def resistance():
    return 100

@pytest.fixture()
def inductance():
    return 1e-6

@pytest.fixture()
def capacitance():
    return 100e-12

@pytest.fixture()
def frequencies():
    return np.logspace(-12,12,1000)

@pytest.fixture()
def rlc_series(voltage, resistance, inductance, capacitance):
    netlist = clb.Netlist()
    netlist.V([1,0], voltage)
    netlist.L([1,2], inductance)
    netlist.C([2,3], capacitance)
    netlist.R([3,0], resistance)
    return netlist

def test_node_voltages(rlc_series, frequencies):

    # Get theoretical node voltages of the RLC circuit using the calculated impedances
    # of the R, L, and C elements.
    V = rlc_series.elements[0]
    L = rlc_series.elements[1]
    C = rlc_series.elements[2]
    R = rlc_series.elements[3]
    z_l = L._impedance(freq=frequencies)
    z_c = C._impedance(freq=frequencies)
    z_r = R._impedance(freq=frequencies)
    v_n1 = V.value
    v_n2 = V.value * ((z_c + z_r) / (z_l + z_c + z_r))
    v_n3 = V.value * (z_r / (z_l + z_c + z_r))

    # Simulate the frequency response of the circuit using MNA and compare node
    # voltages to theoretical values.
    analysis = clb.ModifiedNodalAnalysis(rlc_series, freq=frequencies)

    # what are good parameters for `allclose`?
    assert np.allclose(v_n1, analysis.voltage(1, 'frequency'))
    assert np.allclose(v_n2, analysis.voltage(2, 'frequency'))
    assert np.allclose(v_n3, analysis.voltage(3, 'frequency'))


@pytest.fixture()
def rlc_parallel(current, resistance, inductance, capacitance):
    netlist = clb.Netlist()
    netlist.I([1,0], current)
    netlist.L([0,1], inductance, name='L1', add_kvl=True)
    netlist.C([0,1], capacitance, name='C2', add_kvl=True)
    netlist.R([0,1], resistance, name='R3', add_kvl=True)
    return netlist

def parallel_impedance(*impedances):
    return 1 / sum(1/z for z in impedances)

def test_node_currents(rlc_parallel, frequencies):

    analysis = clb.ModifiedNodalAnalysis(rlc_parallel, freq=frequencies)

    # Get theoretical branch currents
    I = rlc_parallel.elements[0]
    L = rlc_parallel.elements[1]
    C = rlc_parallel.elements[2]
    R = rlc_parallel.elements[3]
    a_l = 1 / L._impedance(freq=frequencies)
    a_c = 1 / C._impedance(freq=frequencies)
    a_r = 1 / R._impedance(freq=frequencies)
    z = 1 / (a_l + a_c + a_r)
    i_l = I.value * z * a_l
    i_c = I.value * z * a_c
    i_r = I.value * z * a_r

    assert np.allclose(i_l, analysis.current('L1', 'frequency'))
    assert np.allclose(i_c, analysis.current('C2', 'frequency'))
    assert np.allclose(i_r, analysis.current('R3', 'frequency'))
