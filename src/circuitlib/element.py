from typing import Tuple, List
from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from .signals import signals

# TODO: should this belong somewhere else?
prefix_symbols = {
    -15: 'f',
    -12: 'p',
    -9: 'n',
    -6: 'µ',
    -3: 'm',
    0: '',
    3: 'k',
    6: 'M',
    9: 'G'
}

class Element(ABC):
    # TODO: see if n_compoments is set to 0 when you create another netlist
    n_elements: int = 0  # I do this for correct naming of elements if no name given, though this could be delegated to Netlist still...
    element_type: str
    element_type_long: str
    si_unit: str

    def __init__(self, nodes, value, name, kvl_idx):
        self.name = name
        self.nodes = nodes
        self.value = value
        # All elements whose currents are to be eliminated are referred to as being
        # in group 1, while all other elements are referred to as group 2. Elements
        # in group 1 are either resistors, independent current sources, VCCS, or
        # CCCS.

        # Specifies the element adds a KVL equation to the MNA matrix with a
        # branch current variable
        self.add_kvl = False if kvl_idx is None else True
        self.kvl_idx = kvl_idx

    # TODO need better param name than `side`!
    @abstractmethod
    def stamps(self):
        return

    @property
    def _formatted_value(self):
        """Returns a string representation of a elements value using the appropriate
        SI units."""

        log_val = np.log10(self.value)
        si_order_of_magnitude = 3 * (log_val // 3)
        mantissa = self.value * 10**(-si_order_of_magnitude)

        return f'{mantissa:.2f} {prefix_symbols[si_order_of_magnitude]}{self.si_unit}'

    def __repr__(self):
        return f'{self.element_type_long}({self.nodes}, {self._formatted_value}, {self.name})'

class Resistor(Element):
    element_type = 'R'
    element_type_long = 'Resistor'
    si_unit = 'Ω'

    def stamps(self):

        if not self.add_kvl:
            return self._group1_stamps()

        return self._group2_stamps()

    def _group1_stamps(self):

        n = self.nodes

        # TODO: work out where to describe the formatting of the sparse matrices
        # Maybe having types would be useful here as a means of documentation
        A = (
                (
                    (1/self.value, -1/self.value, -1/self.value, 1/self.value),
                    ((n[0], n[0], n[1], n[1]), (n[0], n[1], n[0], n[1]))
                ),
                None
        )
        B = C = D = S1 = S2 = (None,)

        return A, B, C, D, S1, S2

    def _group2_stamps(self):

        n = self.nodes

        A = (None, None)
        B = (
                (
                    (1, -1),
                    ((n[0], n[1]), (self.kvl_idx, self.kvl_idx))
                ),
                None
        )
        C = (
                (
                    (1, -1),
                    ((self.kvl_idx, self.kvl_idx), (n[0], n[1]))
                ),
                None
        )
        D = (
                (
                    (-self.value,),
                    ((self.kvl_idx,), (self.kvl_idx,))
                ),
                None
        )
        S1 = (None,)
        S2 = (None,)

        return A, B, C, D, S1, S2

    def _impedance(self, freq=None):
        # For testing
        # Do I need to include freq here?
        # Should this be a complex number?
        if freq is None:
            return self.value
        return np.full_like(freq, self.value)

class Capacitor(Element):
    element_type = 'C'
    element_type_long = 'Capacitor'
    si_unit = 'F'

    def _impedance(self, freq):
        return 1 / (2j * np.pi * freq * self.value)

    def stamps(self):

        if not self.add_kvl:
            return self._group1_stamps()

        return self._group2_stamps()

    def _group1_stamps(self):

        n = self.nodes

        # TODO: work out where to describe the formatting of the sparse matrices
        # Maybe having types would be useful here as a means of documentation
        A = (
                None,
                (
                    (self.value, -self.value, -self.value, self.value),
                    ((n[0], n[0], n[1], n[1]), (n[0], n[1], n[0], n[1]))
                )
        )
        B = C = D = S1 = S2 = (None,)

        return A, B, C, D, S1, S2

    def _group2_stamps(self):

        n = self.nodes

        A = (None, None)
        B = (
                (
                    (1, -1),
                    ((n[0], n[1]), (self.kvl_idx, self.kvl_idx))
                ),
                None
        )
        C = (
                None,
                (
                    (-self.value, self.value),
                    ((self.kvl_idx, self.kvl_idx), (n[0], n[1]))
                )
        )
        D = (
                (
                    (1,),
                    ((self.kvl_idx,), (self.kvl_idx,))
                ),
                None
        )
        S1 = (None,)
        S2 = (None,)

        return A, B, C, D, S1, S2

class Inductor(Element):
    element_type = 'L'
    element_type_long = 'Inductor'
    si_unit = 'H'

    def _impedance(self, freq):
        return 2j * np.pi * freq * self.value

    def stamps(self):

        return self._group2_stamps()

    def _group2_stamps(self):

        n = self.nodes

        A = (None, None)
        B = (
                (
                    (1, -1),
                    ((n[0], n[1]), (self.kvl_idx, self.kvl_idx))
                ),
                None
        )
        C = (
                (
                    (1, -1),
                    ((self.kvl_idx, self.kvl_idx), (n[0], n[1]))
                ),
                None
        )
        D = (
                None,
                (
                    (-self.value,),
                    ((self.kvl_idx,), (self.kvl_idx,))
                )
        )
        S1 = (None,)
        S2 = (None,)

        return A, B, C, D, S1, S2


class VoltageSource(Element):
    element_type = 'V'
    element_type_long = 'Voltage Supply'
    si_unit = 'V'

    def __init__(self, nodes, value, name, kvl_idx, signal, **signal_kwargs):
        # TODO: put in exception if kvl_idx is None for voltage source
        super().__init__(nodes, value, name, kvl_idx)
        self.signal = self._initialize_signal(signal, value, signal_kwargs)
        self.signal_kwargs = signal_kwargs

    def _initialize_signal(self, signal, value, signal_kwargs):
        # For static analysis
        if signal is None:
            return lambda _: value

        # TODO: For transient analysis
        signal_func = signals[signal]
        return signal_func
        # initialized_signal = partial(signal_func, value=value, **signal_kwargs)
        # return initialized_signal

    def stamps(self):

        return self._group2_stamps()

    def _group2_stamps(self):

        n = self.nodes

        A = (None, None)
        B = (
                (
                    (1, -1),
                    ((n[0], n[1]), (self.kvl_idx, self.kvl_idx))
                ),
                None
        )
        C = (
                (
                    (1, -1),
                    ((self.kvl_idx, self.kvl_idx), (n[0], n[1]))
                ),
                None
        )
        D = (None, None)
        S1 = (None,)
        S2 = ((
                (self.value,),
                ((self.kvl_idx,), (0,))
        ),)

        return A, B, C, D, S1, S2

class CurrentSource(Element):
    element_type = 'I'
    element_type_long = 'Current Supply'
    si_unit = 'I'

    def __init__(self, nodes, value, name, kvl_idx, signal, **signal_kwargs):
        # TODO: put in exception if kvl_idx is None for voltage source
        super().__init__(nodes, value, name, kvl_idx)
        self.signal = self._initialize_signal(signal, value, signal_kwargs)

    # TODO: make a Source class that both Current/Voltage sources can inheret from
    # this might extend to current controlled and voltage controlled sources
    def _initialize_signal(self, signal, value, signal_kwargs):
        # For static analysis
        if signal is None:
            return lambda _: value

        # TODO: For transient analysis
        signal_func = signals[signal]
        initialized_signal = partial(signal_func, value=value, **signal_kwargs)
        return initialized_signal

    def stamps(self):

        if not self.add_kvl:
            return self._group1_stamps()

        return self._group2_stamps()

    def _group1_stamps(self):

        n = self.nodes

        A = B = C = D = S2 = (None,)

        S1 = ((
                (-self.value, self.value),
                ((n[0],n[1]), (0,0))
        ),)

        return A, B, C, D, S1, S2

    def _group2_stamps(self):

        n = self.nodes

        A = (None, None)
        B = (
                (
                    (1, -1),
                    ((n[0], n[1]), (self.kvl_idx, self.kvl_idx))
                ),
                None
        )
        C = (None,)
        D = (
                (
                    (1,),
                    ((self.kvl_idx,), (self.kvl_idx,))
                ),
                None
        )

        S1 = (None,)
        S2 = ((
                (self.value,),
                ((self.kvl_idx,), (0,))
        ),)

        return A, B, C, D, S1, S2
