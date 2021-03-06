.. _circuits:

Building Circuits
==============================



Circuits are defined in simpycirc as callable functions, making it easier to simulate the response of a given circuit for different component values. These circuits are generated using the :func:`circuitlib.circuit.simulate` function, which can be called in two manners: writing circuits as text and defining netlists.

Circuit components
-------------------------
Currently, circuitlib only supports linear circuits using a voltage source, meaning circuits can only comprise of resistors, capacitors and inductors. Current sources are also not supported yet.

Each component in a circuit must be given a unique name. The first character of this name is used as an identifier for what type of component it represents

* R = resistor
* C = capacitor
* L = inductor

Otherwise, you can name these components whatever you like, though as these are still defined as variables within the python scope, names must still adhere to the python variable naming conventions (e.g. no hyphens). Two suitable naming conventions are to simply number each component, though a more descriptive naming convention may make it easier to remember the purpose of each component.

.. code:: python

    # Numeric naming convention

    R1  # Resistor
    C1  # Capacitor
    L1  # Inductor

    # Alternative naming convention

    R_ground
    C_smoothing
    L_choke



Frequency range
-------------------------
Typically simulations of electrical circuits are performed in either the frequency or time domain. Only frequency domain simulations are supporting at the moment. Therefore, it is useful to define the frequency of interest before simulating your circuit. By default a frequency of 1000 Hz is given, by a frequency range, defined by a numpy array, can also be supplied to the simulating function, which can be noted in the remaining examples on this page.



Writing circuits by hand
-------------------------
For relatively simple circuits, it may feel more intuitive to simply write out the circuit, similar to how one might with a pen and paper. To achieve this, define the circuit within a function and decorate it with :func:`simpycirc.circuit.simulate`. This magically identifies the circuit components and builds the circuit into a callable function. The circuit can be simulated by calling the function like any other function, with the necessary component values passed as args or kwargs.


.. code:: python

    import numpy as np
    from simpycirc.figures import bode
    from simpycirc.circuit import simulator

    freq = np.logspace(-2,5,100)

    @simulator(freq=freq)
    def my_circuit():
        return R1 + (R2 | C1)

    V1 = my_circuit(R1=100, R2=200, C1=100e-9)
    V2 = my_circuit(R1=100, R2=100, C1=200e-9)

    fig, ax = plt.subplots()
    bode(V1, V2, freq=freq)



Default values can be defined the same as for a regular function.


.. code:: python

    @simulator(freq=freq)
    def my_circuit(R1=100, R2=200, C1=100e-9):
        return R1 + (R2 | C1)

    V1 = my_circuit()


Defining netlists
-----------------

For more verbose circuits, it is recommended to define a netlist, which is given by the :func:`simpycirc.utils.Netlist` object. This may feel more familiar for users already well acquainted with SPICE. Each circuit component is defined by their node connections and their value, and is given as an attribute to the :func:`Netlist` object.

.. code:: python

    from simpycirc.utils import Netlist
    from simpycirc.circuit import simulator

    nl = Netlist()

    nl.R1 = (1, 2), 100
    nl.R2 = (2, 3), 200
    nl.C1 = (2, 3), 100e-9

    my_circuit = simulator(nl)



A note on changing values
--------------------------
