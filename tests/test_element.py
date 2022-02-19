import pytest

from circuitlib.element import Resistor

@pytest.fixture(params=[0.1, 10, 1000])
def resistor_value(request):
    return request.param

@pytest.fixture(params=[None,0])
def resistor_kvl_index(request):
    return request.param

@pytest.fixture(params=[[0,1]])
def nodes(request):
    return request.param

# TODO: will need to add in checks that name is a string (I think...)
@pytest.fixture(params=['r1', None, 123])
def name(request):
    return request.param

@pytest.fixture()
def simple_resistor():
    return Resistor([0,1], 100, 'r1', None)

@pytest.fixture()
def resistor(nodes, resistor_value, name, resistor_kvl_index):
    return Resistor(nodes, resistor_value, name, resistor_kvl_index)

def test_resistor_nodes(resistor, nodes):
    assert resistor.nodes == nodes

def test_resistor_name(resistor, name):
    assert resistor.name == name

def test_resistor_kvl_idx(resistor, resistor_kvl_index):
    assert resistor.kvl_idx == resistor_kvl_index

def test_resistor_add_kvl(resistor, resistor_kvl_index):
    if resistor_kvl_index is None:
        assert resistor.add_kvl == False
    else:
        assert resistor.add_kvl == True

def test_resistor_element_type(simple_resistor):
    assert simple_resistor.element_type == 'R'

def test_resistor_element_type_long(simple_resistor):
    assert simple_resistor.element_type_long == 'Resistor'

def test_resistor_si_unit(simple_resistor):
    assert simple_resistor.si_unit == 'Ω'

def test_A_stamp(resistor):

    A = resistor.stamps()[0]

    if not resistor.add_kvl:

        n0, n1 = resistor.nodes
        val = 1/resistor.value

        data = (val, -val, -val, val)
        row = (n0, n0, n1, n1)
        col = (n0, n1, n0, n1)

        assert A[0] == (data, (row, col))
    else:
        assert A[0] is None

def test_B_stamp(resistor):

    B = resistor.stamps()[1]

    if not resistor.add_kvl:
        assert B[0] is None
    else:
        n0, n1 = resistor.nodes

        data = (1, -1)
        row = (n0, n1)
        col = (resistor.kvl_idx, resistor.kvl_idx)

        assert B[0] == (data, (row, col))

def test_C_stamp(resistor):

    C = resistor.stamps()[2]

    if not resistor.add_kvl:
        assert C[0] is None
    else:
        n0, n1 = resistor.nodes

        data = (1, -1)
        row = (resistor.kvl_idx, resistor.kvl_idx)
        col = (n0, n1)

        assert C[0] == (data, (row, col))

def test_D_stamp(resistor):

    D = resistor.stamps()[3]

    if not resistor.add_kvl:
        assert D[0] is None
    else:
        n0, n1 = resistor.nodes

        data = (-resistor.value,)
        row = (resistor.kvl_idx,)
        col = (resistor.kvl_idx,)

        assert D[0] == (data, (row, col))

def test_S1_stamp(resistor):
    S1 = resistor.stamps()[4]
    assert S1[0] is None

def test_S2_stamp(resistor):
    S2 = resistor.stamps()[5]
    assert S2[0] is None

# TODO: is there a way to automate this parameterisation or must we hardcode
# examples?
@pytest.mark.parametrize('value, formatted_value',
        [(0.01, '10.00 mΩ'), (100, '100.00 Ω'), (1.56e5, '156.00 kΩ'), (12e10, '120.00 GΩ')]
)
def test_resistor_value_formatting(value, formatted_value):
    resistor = Resistor([0,1], value, 'r1', None)
    assert resistor._formatted_value == formatted_value
