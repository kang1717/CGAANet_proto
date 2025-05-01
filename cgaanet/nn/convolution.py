from typing import List, Callable

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import Irreps, TensorProduct, Linear, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from .equivariant_gate import EquivariantGate

import cgaanet._keys as KEY
from cgaanet._const import AtomGraphDataType

from .activation import ShiftedSoftPlus
from .util import broadcast


def message_gather(
    node_features: torch.Tensor,
    edge_dst: torch.Tensor,
    message: torch.Tensor
):
    index = broadcast(edge_dst, message, 0)
    out_shape = [len(node_features)] + list(message.shape[1:])
    out = torch.zeros(
        out_shape,
        dtype=node_features.dtype,
        device=node_features.device
    )
    out.scatter_reduce_(0, index, message, reduce='sum')
    return out


@compile_mode('script')
class IrrepsConvolution(nn.Module):
    """
    convolution of (fig 2.b), comm. in LAMMPS
    """

    def __init__(
        self,
        act_scalar_dict: Callable,
        act_gate_dict: Callable,
        irreps_node_attr: Irreps,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        is_parallel: bool = False,
        data_key_operand: str = KEY.NODE_ATTR,
    ):
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx
        self.is_parallel = is_parallel

        instructions = []
        irreps_mid = []
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
        self.convolution = TensorProduct(
            irreps_x,
            irreps_filter,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_layer_act = weight_layer_act

        self.weight_nn = FullyConnectedNet(
            weight_layer_input_to_hidden + [self.convolution.weight_numel],
            weight_layer_act,
        )
        self._comm_size = self.convolution.irreps_in1.dim  # used in parallel

        ####################### gate ####################

        self.key_operand = data_key_operand

        parity_mapper = {'e': 1, 'o': -1}
        act_scalar_dict = {
            parity_mapper[k]: v for k, v in act_scalar_dict.items()
        }
        act_gate_dict = {parity_mapper[k]: v for k, v in act_gate_dict.items()}

        irreps_gated_elem = []
        irreps_scalars_elem = []
        # non scalar irreps > gated / scalar irreps > scalars

        #print (irreps_x)
        for mul, irreps in irreps_mid:
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

        irreps_in = self.gate.irreps_in

        self.self_interaction = Linear(irreps_mid,irreps_in)

        self.irreps_mid = irreps_mid
        self.irreps_node_attr = irreps_node_attr

        self.fc_tensor_product = FullyConnectedTensorProduct(
            irreps_x, irreps_x, irreps_x
        #    irreps_mid, irreps_mid, irreps_in
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        weight = self.weight_nn(data[self.key_weight_input])
        x = data[self.key_x]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        # note that 1 -> src 0 -> dst
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        tensor_product = self.fc_tensor_product(x[edge_src],x[edge_dst]) 

        message = self.convolution(tensor_product, data[self.key_filter], weight)

        message = self.self_interaction(message) 

        message = self.gate(message)

        x = message_gather(x, edge_dst, message)
        x = x.div(self.denominator)
        if self.is_parallel:
            # NLOCAL is # of atoms in system at 'CPU'
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data
