from typing import Callable, List, Optional, Dict

import torch
import torch.nn as nn

from e3nn.util.jit import compile_mode
from e3nn.o3 import Irreps, Linear
from e3nn.nn import Gate

import cgaanet._keys as KEY
from cgaanet._const import AtomGraphDataType

from .equivariant_gate import EquivariantGate
from .linear import IrrepsLinear

import numpy as np

import time
@compile_mode('script')
class GrainEncoding(nn.Module):
    """
    Encodes grain information into a unified tensor with 0e and 1o irreps.

    Args:
        total_irreps: Total size of 0e and 1o irreps across all grain types.
        grain_atom_counts: A dictionary mapping grain types to their atom counts.
        data_key_type: Key for accessing grain types in the data.
        data_key_relative_pos: Key for accessing relative positions in the data.
    """

    def __init__(
        self,
        grain_atom_counts: dict,
        device: str, 
        data_key_type: str = KEY.NODE_FEATURE,
        data_key_relative_pos: str = KEY.RELATIVE_POS,
        data_key_zeros: str = KEY.ZERO_ARRAY,
    ):
        super().__init__()
        self.data_key_type = data_key_type
        self.data_key_relative_pos = data_key_relative_pos
        self.key_zeros = data_key_zeros
        self.device = device
        self.grain_atom_counts = grain_atom_counts

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:

        #data[self.data_key_relative_pos].requires_grad_(True)

        data[self.data_key_type] = torch.cat((data[self.key_zeros],data[self.data_key_relative_pos]),dim=1)


        return data


@compile_mode('script')
class IrrepsLinear_grain_embedding(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        data_key_additional: str = KEY.NODE_ATTR,  # additional output
        **e3nn_linear_params,
    ):
        super().__init__()
        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        self.key_additional_output = data_key_additional

        self.linear = Linear(irreps_in, irreps_out, **e3nn_linear_params)


    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:

        data[self.key_output] = self.linear(data[self.key_input])


        return data

def IntraGrainActivation(
        t: int,
        irreps_x: Irreps,
        act_scalar_dict: Dict[int, Callable],
        act_gate_dict: Dict[int, Callable],
        biases: str,
        data_key_x: str = KEY.NODE_FEATURE,
    ):

        block = {}

        gate = EquivariantGate(irreps_x, act_scalar_dict, act_gate_dict)
        
        irreps_for_gate_in = gate.get_gate_irreps_in()

        block[f'{t}_intra_grain_linear'] = IrrepsLinear(
            irreps_x,
            irreps_for_gate_in,
            data_key_in = data_key_x,
            biases=biases,
        )

        block[f'{t}_intra_grain_gate'] = gate


        return block
 
@compile_mode('script')
class save_additional_key(nn.Module):
    def __init__(
        self,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_additional: str = KEY.NODE_ATTR,
    ):
        super().__init__()
        self.key_x = data_key_x
        self.key_additional_output = data_key_additional

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:

        data[self.key_additional_output] =  data[self.key_x].clone() # for self interaction


        return data

@compile_mode('script')
class split_energy_force_line(nn.Module):

    def __init__(
        self, 
        irreps_in: str,
        key_x: str = KEY.NODE_FEATURE,
        key_f: str = KEY.FORCE_TMP,
    ):

        super().__init__()
        self.irreps_in = irreps_in
        self.key_x = key_x
        self.key_f = key_f

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:

        offset = 0
        data_0e_list = []
        data_1o_list = []

        for (mul, ir) in self.irreps_in:
            chunk_size = mul * ir.dim 
            chunk = data[self.key_x][..., offset : offset + chunk_size]  
            offset += chunk_size

            if ir.l == 0 and ir.p == 1:
                data_0e_list.append(chunk)
            elif ir.l == 1 and ir.p == -1:
                data_1o_list.append(chunk)
            else:
                pass

        data_0e = torch.cat(data_0e_list, dim=-1) if len(data_0e_list) > 0 else None
        data_1o = torch.cat(data_1o_list, dim=-1) if len(data_1o_list) > 0 else None

        data[self.key_x] = data_0e
        data[self.key_f] = data_1o 

        return data


@compile_mode('script')
class make_average_zero(nn.Module):

    def __init__(
        self, 
        grain_atom_counts: dict,
        device: str,
        key_numbers: str = KEY.ATOMIC_NUMBERS,
        key_forces: str = KEY.INTRA_GRAIN_FORCES
    ):

        super().__init__()
        self.grain_atom_counts = grain_atom_counts
        self.key_numbers = key_numbers
        self.key_forces = key_forces
        self.device = device

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:


        numbers = data[self.key_numbers].flatten()

        unique_types = sorted(self.grain_atom_counts.keys())

        running_sum = 0
        offsets = {}
        for t in unique_types:
            offsets[t] = running_sum
            running_sum = running_sum + self.grain_atom_counts[t]

        # Create a zero-initialized tensor of the same shape
        rearranged_forces = torch.zeros_like(data[self.key_forces])  # (N, 3M)


        for t in unique_types:
            mask = (numbers == t)  # shape (N,) boolean
            num_t = self.grain_atom_counts[t]
    
            start_col = offsets[t] * 3
            end_col   = start_col + (num_t * 3)
    
            # Copy the first (num_t*3) columns from the original 'forces'
            # into [start_col:end_col] in the new tensor
            rearranged_forces[mask, start_col:end_col] = data[self.key_forces][mask, : (num_t * 3)]

            # sub_forces shape: (#rows_of_type_t, 3*num_t)
            sub_forces = rearranged_forces[mask, start_col:end_col]
    
            if sub_forces.numel() == 0:
                continue

            # reshape: (#rows_of_type_t, num_t, 3)
            sub_forces_3d = sub_forces.reshape(-1, num_t, 3)

            #    shape = (#rows_of_type_t, 3)
            row_mean = sub_forces_3d.mean(dim=1, keepdim=True)

            sub_forces_3d = sub_forces_3d - row_mean 

            #    (#rows_of_type_t, 3*num_t)
            sub_forces_updated = sub_forces_3d.reshape(sub_forces.shape)
            rearranged_forces[mask, start_col:end_col] = sub_forces_updated


        data[self.key_forces] = rearranged_forces



        return data


@compile_mode('script')
class ForceOutputGrain(nn.Module):
    """
    Compute stress and force from edge.
    Used in parallel torchscipt models, and training
    """
    def __init__(
        self,
        data_key_pos: str = KEY.RELATIVE_POS,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_force: str = KEY.INTRA_GRAIN_FORCES,
    ):

        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self._is_batch_data = True

    def get_grad_key(self):
        return self.key_edge

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:


        relative_pos = data[self.key_pos]
        energy = [(data[self.key_energy]).sum()]

        #data[KEY.EDGE_VEC].requires_grad_(False)
        #data[KEY.RELATIVE_POS].requires_grad_(True)

        grad = torch.autograd.grad(
            energy,
            [relative_pos],
            create_graph=self.training,
            allow_unused=True
        )


        # make grad is not Optional[Tensor]
        data[self.key_force] = grad[0]



        return data

def repeat_triplet_conditional(tensor1, tensor2):
    mask = tensor2.unsqueeze(-1).float()  # (batch_size, N, 1)
    
    tensor1_expanded = tensor1.unsqueeze(1).repeat(1, tensor2.size(1), 1)  # (batch_size, N, 3)
    
    output = tensor1_expanded * mask  # (batch_size, N, 3)
    
    final_output = output.view(tensor1.size(0), -1)  # (batch_size, N*3)
    
    return final_output


@compile_mode('script')
class make_zero(nn.Module):

    def __init__(
        self, 
        grain_atom_counts: dict,
        device: str,
        key_numbers: str = KEY.ATOMIC_NUMBERS,
        key_forces: str = KEY.INTRA_GRAIN_FORCES,
        key_zeros: str = KEY.TRIPLE_ZERO,
        key_zeros_x: str = KEY.ZEROS_X,
        key_zeros_y: str = KEY.ZEROS_Y,
        key_zeros_z: str = KEY.ZEROS_Z,
    ):

        super().__init__()
        self.grain_atom_counts = grain_atom_counts
        self.key_numbers = key_numbers
        self.key_forces = key_forces
        self.device = device
        self.key_zeros = key_zeros
        self.key_zeros_x = key_zeros_x
        self.key_zeros_y = key_zeros_y
        self.key_zeros_z = key_zeros_z

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:

        data[self.key_forces] = data[self.key_forces] * data[self.key_zeros] 

        natom_row = data[self.key_zeros_x].sum(dim=1).unsqueeze(1)

        force_x = data[self.key_forces]*data[self.key_zeros_x]
        force_y = data[self.key_forces]*data[self.key_zeros_y]
        force_z = data[self.key_forces]*data[self.key_zeros_z]

        mean_force_x = force_x.sum(dim=1).unsqueeze(1) / natom_row * data[self.key_zeros_x] 
        mean_force_y = force_y.sum(dim=1).unsqueeze(1) / natom_row * data[self.key_zeros_y] 
        mean_force_z = force_z.sum(dim=1).unsqueeze(1) / natom_row * data[self.key_zeros_z] 

        data[self.key_forces] = data[self.key_forces] - mean_force_x - mean_force_y - mean_force_z

        return data

@compile_mode('script')
class MergeForces(nn.Module):
    """
    Compute stress and force from edge.
    Used in parallel torchscipt models, and training
    """
    def __init__(
        self,
        data_key_zero: str = KEY.ZERO_ARRAY,
        data_key_mean_force: str = KEY.PRED_FORCE,
        data_key_force: str = KEY.INTRA_GRAIN_FORCES,
        data_key_total_force: str= KEY.TOT_FORCE_PRED,
    ):

        super().__init__()
        self.key_zero = data_key_zero
        self.key_mean_force = data_key_mean_force
        self.key_force = data_key_force
        self.key_total_force = data_key_total_force
 
    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:


        natom_row = data[self.key_zero].sum(dim=1).unsqueeze(1)

        mean_force = data[self.key_mean_force] / natom_row   # inter- force divided by the number of atoms
        
        mean_force_modified = repeat_triplet_conditional(mean_force,data[self.key_zero])   # inter_Force preprocess

        data[self.key_total_force] = data[self.key_force] + mean_force_modified

        return data
