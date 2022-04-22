import pytest

from circuitlib.element import Resistor, VoltageSource

@pytest.fixture()
def resistor():
    resistor = Resistor([0,1], 100, 'r1', None)
    return resistor

@pytest.fixture()
def resistor_group1_stamps(resistor):
    return resistor.stamps()

@pytest.fixture()
def static_voltage():
    return VoltageSupply([0,1], 5, name='v1', kvl_idx=0, signal=None)

