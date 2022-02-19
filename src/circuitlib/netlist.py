from typing import List, Set
from . import element

import numpy as np
from scipy import sparse

class Netlist:

    def __init__(self):
        self.nodes: Set[int] = set()
        self.branch_idxs = {}
        self._elements: List[element.Element] = []
        self.n_kvl_eqs: int = 0
        # TODO: every time we add a new node we should add a large resistance
        # to ground to aid in DC analysis (p.133).

    # allow setting of elements directly, mainly useful in testing as it allows
    # use to programatically set up a netlist without having to use the API
    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, elements):

        nodes = set()
        n_kvl_eqs = 0

        for element in elements:

            nodes.update(element.nodes)
            if element.kvl_idx is not None:
                n_kvl_eqs += 1

        self.n_kvl_eqs = n_kvl_eqs
        self.nodes = nodes
        self._elements = elements

    # TODO: core feature - test thoroughly
    # TODO: I don't like the `add_kvl` param name. Maybe `solve_current` or
    # `measure_current`? Or maybe `add_branch_current`? So for each circuit
    # we need to keep track of the nodes and the branches.
    def R(self, nodes, value, name=None, add_kvl=False):
        """Linear resistor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals
            of the component. value (float): Value of the component in SI units.
        """

        self.nodes.update(nodes)  # TODO: unsure if this is the best way to keep track of nodes

        name = name or f"R{len(self.elements)}"
        resistor = element.Resistor(
                nodes=nodes,
                value=value,
                name=name,
                kvl_idx=self.n_kvl_eqs if add_kvl else None  # TODO: is this passing in a copy?
        )

        # TODO: this pattern will be repeased in a lot of places, could we turn
        # this into a decorator?
        if add_kvl:
            self.branch_idxs[name] = self.n_kvl_eqs
            self.n_kvl_eqs += 1

        self.elements.append(resistor)

        # TODO: returning the comopnent may allow you to pass it to other components?
        # This could be useful if you want to specify that the output of other
        # components is dependent on this one (e.g. CCVC). If not then could
        # remove this return statement.
        return resistor

    def C(self, nodes, value, name=None, add_kvl=False):
        """Linear capacitor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals
            of the component. value (float): Value of the component in SI units.
        """

        name = name or f"C{len(self.elements)}"
        capacitor = element.Capacitor(
                nodes=nodes,
                value=value,
                name=name,
                kvl_idx=self.n_kvl_eqs if add_kvl else None  # TODO: is this passing in a copy?
        )

        # TODO: this pattern will be repeased in a lot of places, could we turn
        # this into a decorator?
        self.nodes.update(nodes)  # TODO: unsure if this is the best way to keep track of nodes
        if add_kvl:
            self.branch_idxs[name] = self.n_kvl_eqs
            self.n_kvl_eqs += 1
        self.elements.append(capacitor)

        # TODO: returning the comopnent may allow you to pass it to other components?
        # This could be useful if you want to specify that the output of other
        # components is dependent on this one (e.g. CCVC). If not then could
        # remove this return statement.
        return capacitor

    def L(self, nodes, value, name=None, add_kvl=False):
        """Linear inductor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals
            of the component. value (float): Value of the component in SI units.
        """

        name = name or f"L{len(self.elements)}"
        inductor = element.Inductor(
                nodes=nodes,
                value=value,
                name=name,
                kvl_idx=self.n_kvl_eqs
        )

        self.nodes.update(nodes)
        self.branch_idxs[name] = self.n_kvl_eqs
        self.n_kvl_eqs += 1
        self.elements.append(inductor)

        return inductor

    def V(self, nodes, value, name=None, signal=None, **signal_kwargs):
        """Voltage source.

        Args:
            nodes (list): Nodes connected the positive and negative terminals
            of the component. value (float): Value of the component in SI units.
        """

        name = name or f"V{len(self.elements)}"
        voltage = element.VoltageSource(
                nodes=nodes,
                value=value,
                name=name,
                kvl_idx=self.n_kvl_eqs,
                signal=signal,
                **signal_kwargs
        )

        self.nodes.update(nodes)
        self.branch_idxs[name] = self.n_kvl_eqs
        self.n_kvl_eqs += 1
        self.elements.append(voltage)

        return voltage

    def I(self, nodes, value, name=None, add_kvl=False, signal=None, **signal_kwargs):
        """Current source.

        Args:
            nodes (list): Nodes connected the positive and negative terminals
            of the component. value (float): Value of the component in SI units.
        """

        name = name or f"I{len(self.elements)}"
        current = element.CurrentSource(
                nodes=nodes,
                value=value,
                name=name,
                kvl_idx=self.n_kvl_eqs if add_kvl else None,  # TODO: is this passing in a copy?
                signal=signal,
                **signal_kwargs
        )

        self.nodes.update(nodes)
        if add_kvl:
            self.branch_idxs[name] = self.n_kvl_eqs
            self.n_kvl_eqs += 1
        self.elements.append(current)

        return current

    def __len__(self):
        return len(self.nodes)

    # TODO: core feature - test thoroughly
    # TODO: better name for this function?
    def matrix(self):
        '''Have some good doc strings showing what ABCDS1S2 are?

        Could do like below or could render them in LaTex.
        ┌     ┐┌   ┐   ┌    ┐
        │ A B ││ x │ = │ S1 │
        │ C D ││ y │   │ S2 │
        └     ┘└   ┘   └    ┘

        Each A, B, C, D matrix is composed of two parts, the static and dynamic
        elements. 
        '''

        A, B, C, D, S1, S2 = self._initialize_matrixes()

        for element in self.elements:

            A, B, C, D, S1, S2 = self._stamp_matrixes(
                element.stamps(),
                [A, B, C, D, S1, S2],
            )

        # TODO: come up with simple naming convention for the lhs matrixes...
        # e.g. lhs_static, lhs_dynamic
        lhs_static, lhs_dynamic = zip(A, B, C, D)

        # TODO: whats a more correct name for these matrices?
        reduced_indicence_matrix_static = self._build_lhs_matrixes(*lhs_static)
        reduced_indicence_matrix_dynamic = self._build_lhs_matrixes(*lhs_dynamic)

        full_source_matrix = sparse.bmat([[S1[0]],[S2[0]]], format='csr')
        reduced_source_matrix = full_source_matrix[1:]

        return (
            reduced_indicence_matrix_static,
            reduced_indicence_matrix_dynamic,
            reduced_source_matrix  # TODO: verify S stays a "csr" matrix
        )

    def _build_lhs_matrixes(self, A, B, C, D):
        full_indicence_matrix = sparse.bmat([[A,B],[C,D]], format='csr')
        reduced_indicence_matrix = full_indicence_matrix[1:,1:]
        return reduced_indicence_matrix

    def _validate_circuit(self):
        # TODO: validate the circuit is correctly set up. e.g. fully connected
        # no self-loops etc.
        pass

    def _initialize_matrixes(self):
        # TODO: there is still something overly verbose about this approach of
        # creating and updating the matrices.
        # I dont like hiding these as elements in a list, with them implictly updated
        # as you iterate through the items in the list.
        # Another approach is to have each subblock as attributes of the class
        # but this could cause issues if you want to update the values for a
        # particular simulation?

        # Dimensions of (2, ...) to handle stamps of dynamic elements (i.e. C and L)
        # where the node current/voltage is dependent on the derivative of voltage/current
        # TODO these are duplicated arrays because we cant have 3d sparse arrays!
        A = [
            sparse.csr_matrix((len(self), len(self))),
            sparse.csr_matrix((len(self), len(self))),
        ]
        B = [
            sparse.csr_matrix((len(self), self.n_kvl_eqs)),
            sparse.csr_matrix((len(self), self.n_kvl_eqs)),
        ]
        C = [
            sparse.csr_matrix((self.n_kvl_eqs, len(self))),
            sparse.csr_matrix((self.n_kvl_eqs, len(self))),
        ]
        D = [
            sparse.csr_matrix((self.n_kvl_eqs, self.n_kvl_eqs)),
            sparse.csr_matrix((self.n_kvl_eqs, self.n_kvl_eqs)),
        ]
        S1 = [
            sparse.csr_matrix((len(self), 1)),
        ]
        S2 = [
            sparse.csr_matrix((self.n_kvl_eqs, 1)),
        ]
        return A, B, C, D, S1, S2


    def _stamp_matrixes(self, element_stamps, matrixes):

        stamped_matrixes = []
        for m, stamps in zip(matrixes, element_stamps):

            for i, s in enumerate(stamps):
                if s is not None:
                    m[i] += sparse.coo_matrix(s, shape=m[i].shape)

            stamped_matrixes.append(m)

        return stamped_matrixes

