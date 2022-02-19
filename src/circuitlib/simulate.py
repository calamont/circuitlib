import warnings
from collections.abc import Iterable

import numpy as np
import scipy.sparse.linalg as sparse

from .solver import dae

class ModifiedNodalAnalysis:

    def __init__(self, netlist, *, freq=None, time=None):
        # TODO: validate netlist is correctly specified
        self.netlist = netlist
        self.freq = self._valitate_array_input(freq, 'freq')
        self.time = self._valitate_array_input(time, 'time')

    def _valitate_array_input(self, array, array_type):

        if array is None:
            return array

        array = np.array(array)

        if array.ndim > 1:
            array = array.flatten()
            warnings.warn(f'Flattening {array_type} as {array_type}.ndim > 1.')

        return array

    def current(self, branch, analysis):
        idx = len(self.netlist) - 1 + self.netlist.branch_idxs[branch]
        response = self.response(analysis=analysis)
        return response[...,idx]

    def voltage(self, node, analysis):

        if node <= 0:
            raise ValueError('Node value must be greater than 0, which is ground.')
        elif node > max(self.netlist.nodes):
            raise ValueError(f'Node {node} does not exist.')

        idx = node - 1
        response = self.response(analysis=analysis)
        return response[...,idx]

    def response(self, analysis):
        if analysis == 'dc':
            # TODO: setting the frequency to 0 may be a bit of a hack but it works
            # Test this rigorously to confirm this works in most scenarios
            return self._frequency_response(freq=np.array([0]))
        elif analysis == 'frequency':
            return self._frequency_response(freq=self.freq)
        # TODO: implement transient analysis
        elif analysis == 'transient':
            return self._transient_response(time=self.time)

    def _frequency_response(self, freq):
        rim_static, rim_dynamic, rsm = self.netlist.matrix()

        # TODO: converting sparse matrixes to dense to allow 3 dimensions
        # for frequency dependent impedances
        rim_static = np.array(rim_static.todense())[None,:,:]
        rim_dynamic = np.array(rim_dynamic.todense())[None,:,:]
        rsm = np.array(rsm.todense())[None,:]
        freq = freq[:,None,None]

        rim = (
                rim_static
                + 1j * 2 * np.pi * freq * rim_dynamic
        )
        node_values = np.linalg.solve(rim, rsm)
        # TODO: can we broadcast the numpy arrays better so we aren't left over
        # with a 3rd dimension of size 1?
        return np.squeeze(node_values, axis=-1)

    def _transient_response(self, time):

        rim_static, rim_dynamic, rsm = self.netlist.matrix()

        # TODO: converting sparse matrixes to dense to allow 3 dimensions
        # for frequency dependent impedances
        rim_static = np.array(rim_static.todense(), dtype=np.double, order='F')
        rim_dynamic = np.array(rim_dynamic.todense(), dtype=np.double, order='F')
        rsm = np.array(rsm.todense(), dtype=np.double, order='F')
        init = self._initial_conditions(rsm)
        components = self._construct_components(self.netlist.elements)
        time = time

        # raise Exception
        node_values = dae(time, rim_static, rim_dynamic*2, init, components)
        # TODO: can we broadcast the numpy arrays better so we aren't left over
        # with a 3rd dimension of size 1?

        return node_values
        # return np.squeeze(node_values, axis=-1)

    def _initial_conditions(self, rsm):
        '''Conveniece function while testing DAE solver'''
        return np.zeros_like(rsm, dtype=np.double, order='F').flatten()

    def _construct_components(self, elements):
        '''Conveniece function while testing DAE solver'''

        kwarg_defaults = {
            "period": 0,
            "x_offset": 0,
            "y_offset": 0,
            "mod": 0.5,
        }
        components = {}
        for e in elements:

            signal_kwargs = {
                **kwarg_defaults,
                **e.__dict__.get('signal_kwargs', {})
            }

            components[e.name] = {
                'nodes': e.nodes,
                'dependent_nodes': None,
                'value': e.value,
                'group2_idx': max(self.netlist.nodes) + e.kvl_idx if e.kvl_idx is not None else 0,
                'type': e.element_type,
                'signal': e.__dict__.get('signal'),
                'set_kwargs': signal_kwargs
            }
        return components

