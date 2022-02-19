import pytest

import numpy as np
import circuitlib as clb

@pytest.fixture()
def voltage():
    return 10

@pytest.fixture()
def current():
    return 10e-3

@pytest.fixture()
def resistance():
    return 250

@pytest.fixture()
def inductance():
    return 10e-3

@pytest.fixture()
def capacitance():
    return 1e-6

@pytest.fixture()
def time():
    return np.linspace(0, 1e-3, 1000)

@pytest.fixture()
def rlc_series(voltage, resistance, inductance, capacitance):
    netlist = clb.Netlist()
    netlist.V([1,0], voltage, signal='DC')
    netlist.L([1,2], inductance)
    netlist.C([2,3], capacitance)
    netlist.R([3,0], resistance)
    return netlist

'''
We will test our RLC circuit using the analytical solution for the transient
response
https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-071j-introduction-to-electronics-signals-and-measurement-spring-2006/lecture-notes/16_transint_rlc2.pdf
http://tuttle.merc.iastate.edu/ee201/topics/capacitors_inductors/RLC_transients.pdf

We can create a fixture to test the response when:
    - underdamped
    - critically damped
    - overdamped
'''
def test_node_voltages(rlc_series, time):

    Vi = 0  # assume an initial voltage of 0
    # Get theoretical node voltages of the RLC circuit using the calculated impedances
    # of the R, L, and C elements.
    V = rlc_series.elements[0]
    L = rlc_series.elements[1]
    C = rlc_series.elements[2]
    R = rlc_series.elements[3]

    damping_rate = R.value / (2 * L.value)
    resonant_freq = 1 / np.sqrt(L.value * C.value)

    # There are three damping responses, depending on the element values:
    # critically damped - damping_rate == natural_freq
    # underdampd - damping_rate < natural_freq
    # overdampd - damping_rate > natural_freq
    # TODO: How can we dynammically generated these three conditions? Do we keep
    # the values for two elements fixed and vary the third? The obvious choice
    # would be to vary the resistor value. As the resistance increases the system
    # becomes overdamped.

    # The complementary solutions to the 2nd order differential equation
    s1 = -damping_rate + np.sqrt(complex(damping_rate**2 - resonant_freq**2))
    s2 = -damping_rate - np.sqrt(complex(damping_rate**2 - resonant_freq**2))

    A = (V.value - Vi) / (s1/s2 - 1)
    B = (V.value - Vi) / (s2/s1 - 1)

    v_c = A * np.exp(s1 * time) + B * np.exp(s2 * time) + V.value
    # TODO: need to calculate resistor and inductor voltages based off capactior

    # Simulate the frequency response of the circuit using MNA and compare node
    # voltages to theoretical values.
    analysis = clb.ModifiedNodalAnalysis(rlc_series, time=time)
    n2_voltage = analysis.voltage(2, 'transient')
    n3_voltage = analysis.voltage(3, 'transient')

    v_sim = n2_voltage - n3_voltage
    # TODO: what are good parameters for `allclose`?
    assert np.allclose(np.abs(v_c), v_sim, atol=1e-1)

