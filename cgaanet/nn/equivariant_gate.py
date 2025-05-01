from typing import Callable, Dict, Optional

import torch.nn as nn
from e3nn.nn import Gate
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import cgaanet._keys as KEY
from cgaanet._const import AtomGraphDataType


@compile_mode('script')
class EquivariantGate(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        act_scalar_dict: Dict[int, Callable],
        act_gate_dict: Dict[int, Callable],
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_additional: str = KEY.NODE_ATTR,
    ):
        super().__init__()
        self.key_x = data_key_x

        self.key_additional_output = data_key_additional

        parity_mapper = {'e': 1, 'o': -1}
        act_scalar_dict = {
            parity_mapper[k]: v for k, v in act_scalar_dict.items()
        }
        act_gate_dict = {parity_mapper[k]: v for k, v in act_gate_dict.items()}

        irreps_gated_elem = []
        irreps_scalars_elem = []
        # non scalar irreps > gated / scalar irreps > scalars

        #print (irreps_x)
        for mul, irreps in irreps_x:
            #print (mul, irreps)
            if irreps.l > 0:
                irreps_gated_elem.append((mul, irreps))
            else:
                irreps_scalars_elem.append((mul, irreps))


        irreps_scalars = Irreps(irreps_scalars_elem)
        irreps_gated = Irreps(irreps_gated_elem)

        irreps_gates_parity = 1 if '0e' in irreps_scalars else -1
        irreps_gates = Irreps(
            [(mul, (0, irreps_gates_parity)) for mul, _ in irreps_gated]
        )

        act_scalars = [act_scalar_dict[p] for _, (_, p) in irreps_scalars]
        act_gates = [act_gate_dict[p] for _, (_, p) in irreps_gates]

        self.gate = Gate(
            irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
        )

    def get_gate_irreps_in(self):
        """
        user must call this function to get proper irreps in for forward
        """
        return self.gate.irreps_in

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_x] = self.gate(data[self.key_x])
        return data
