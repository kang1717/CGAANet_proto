import copy
import os.path
from functools import partial
from itertools import islice
from typing import Callable, List, Optional
import torch
import ase
import ase.io
import numpy as np
import torch.multiprocessing as mp
from ase.io.vasp_parsers.vasp_outcar_parsers import (
    Cell,
    DefaultParsersContainer,
    Energy,
    OutcarChunkParser,
    PositionsAndForces,
    Stress,
    outcarchunks,
)
from ase.neighborlist import primitive_neighbor_list
from ase.utils import string2index
from braceexpand import braceexpand
from tqdm import tqdm

#import _keys as KEY
import cgaanet._keys as KEY
from cgaanet.atom_graph_data import AtomGraphData

from cgaanet.train.dataset import AtomGraphDataset
from collections import defaultdict
import time

import time
import numpy as np
from collections import defaultdict


def find_closest_pbc_image_3d(positions_2d, cell, inv_cell, pbc):
    """
    Vectorized PBC correction for a 3D tensor (G, num_t, 3).
    positions_3d[g,0,:] is the reference for grain g.
    pbc assumed = (True,True,True) for simplicity.
    """
    fractional_coords = np.dot (positions_2d, inv_cell.T)
    ref_frac = fractional_coords[0].copy()
    #fractional_coords = np.einsum('gac,dc->gad',positions_3d, inv_cell.T)
    
    fractional_coords -= ref_frac
    # minimal image
    fractional_coords -= np.round(fractional_coords)
    fractional_coords += ref_frac

    # back to Cartesian
    corrected_2d = np.dot(fractional_coords, cell)
    return corrected_2d

def calculate_grain_properties(atoms, grain_atom_counts):
    """
    Builds 'grains' based on grain_type and grain_num.
    Each grain_num groups certain grains together.
    Each type t has grain_atom_counts[t] atoms per grain.
    
    Returns the same keys as the old function:
      "grain_types", "center_of_mass", "average_force",
      "relative_positions", "relative_forces", "grain_atomic_ids",
      "offsets", "grain_atom_counts"
    """

    t1 = time.time()

    # 1) Basic retrieval
    grain_types_arr = atoms.get_array("grain_type").astype(int)  # (N_atoms,)
    grain_nums_arr  = atoms.get_array("grain_num").astype(int)    # (N_atoms,)
    positions = atoms.get_positions()                            # (N_atoms,3)
    forces    = atoms.get_forces()                               # (N_atoms,3)
    cell      = atoms.get_cell()                                 # (3,3)
    inv_cell  = np.linalg.inv(cell)
    pbc       = atoms.get_pbc()                                  # (bool,bool,bool)

    # 2) Group atoms by (grain_type, grain_num)
    type_num_dict = defaultdict(list)
    N = len(grain_types_arr)
    for i in range(N):
        t = grain_types_arr[i]
        num = grain_nums_arr[i]
        key = (t, num)
        type_num_dict[key].append(i)

    unique_type_num = sorted(type_num_dict.keys())

    # group atoms by type
    type_dict = defaultdict(list)
    N = len(grain_types_arr)
    for i in range(N):
        t = grain_types_arr[i]
        num = grain_nums_arr[i]
        key = (t,num)
        type_dict[t].append(i)

    # 3) Build column offsets (as in old code)
    unique_types = sorted(grain_atom_counts.keys())
    type_num_grains = {}
    for t in unique_types:
        num_t = grain_atom_counts[t]
        n_atoms = len(type_dict[t])
        # number of grains = ceil(n_atoms / num_t)
        G_t = (n_atoms + num_t - 1) // num_t
        type_num_grains[t] = G_t

    total_count = sum(grain_atom_counts.values())

    # 4) Calculate how many "grains" (chunks) per (type, num)
    type_num_num_grains = {}
    for key in unique_type_num:
        t, num = key
        num_t = grain_atom_counts[t]
        n_atoms = len(type_num_dict[key])
        G_t = (n_atoms + num_t - 1) // num_t  # ceiling division
        type_num_num_grains[key] = G_t

    # total grains = sum of G_t
    total_grains = sum(type_num_num_grains[key] for key in unique_type_num)

    # 5) Prepare final arrays
    final_relative_positions = np.zeros((total_grains, 3*total_count), dtype=float)
    final_relative_forces    = np.zeros((total_grains, 3*total_count), dtype=float)
    final_forces             = np.zeros((total_grains, 3*total_count), dtype=float)
    final_grain_atomic_ids   = -1 * np.ones((total_grains, total_count), dtype=int)
    center_of_mass = np.zeros((total_grains, 3), dtype=float)
    average_force  = np.zeros((total_grains, 3), dtype=float)
    total_force    = np.zeros((total_grains, 3), dtype=float)
    grain_types_out = np.zeros(total_grains, dtype=int)

    # 6) Row offset per (type, num)
    offsets = {}
    offsets_return = []
    running_row = 0
    for key in sorted(grain_atom_counts.keys()):
        offsets[key] = running_row
        start = running_row
        running_row += grain_atom_counts[key]
        offsets_return.append((start,running_row))

    # 7) For each (type, num), zero-pad & reshape => (G_t, num_t, 3)
    for i,key in enumerate(unique_type_num):
        t, num = key
        atoms_for_t = type_num_dict[key]
        atoms_for_t.sort()  # optional
        num_t = grain_atom_counts[t]

        row0 = offsets[t]
        row1 = row0 + num_t

        # Fill grain_types_out
        grain_types_out[i] = t

        # positions_t, forces_t => shape (n_atoms, 3)
        positions_t = positions[atoms_for_t]
        forces_t    = forces[atoms_for_t]


        corrected_2d = find_closest_pbc_image_3d(positions_t, cell, inv_cell, pbc)

      
        # center_of_mass, average_force, total_force => shape (G_t,3)
        cm = corrected_2d.mean(axis=0)
        af = forces_t.mean(axis=0)
        tf = forces_t.sum(axis=0)

        center_of_mass [i] = cm
        average_force  [i] = af
        total_force    [i] = tf

        # relative
        rel_pos_2d = corrected_2d - cm
        rel_for_2d = forces_t     - af


        position_final  = rel_pos_2d.reshape(3*num_t)
        force_final     = forces_t.reshape(3*num_t)
        force_rel_final = rel_for_2d.reshape(3*num_t)


        # fill final arrays
        col0 = 3 * offsets[t]
        col1 = col0 + 3*num_t


        final_relative_positions[i, col0:col1] = position_final
        final_forces[i, col0:col1]             = force_final
        final_relative_forces[i, col0:col1]    = force_rel_final

        # fill grain_atomic_ids
        idx_array = -1 * np.ones((total_count,), dtype=int)
        idx_array[offsets[t]:offsets[t]+num_t] = np.array(atoms_for_t,dtype=int)
        final_grain_atomic_ids[i] = idx_array


    t2 = time.time()

    zero_tensor = torch.zeros((len(grain_types_out), sum(grain_atom_counts.values())), dtype=torch.long)
    start_idx = 0
    for grain_type, count in grain_atom_counts.items():

        end_idx = start_idx + count

        mask = torch.tensor(grain_types_out == grain_type) 
        mask_indices = torch.nonzero(mask, as_tuple=True)[0]

        # Fill zero_tensor
        zero_tensor[mask_indices, start_idx:end_idx] = 1.0

        start_idx = end_idx

    mapping_x = torch.tensor([[0,0,0],[1,0,0]], dtype=torch.long)
    mapping_y = torch.tensor([[0,0,0],[0,1,0]], dtype=torch.long)
    mapping_z = torch.tensor([[0,0,0],[0,0,1]], dtype=torch.long)
   
    transformed_3d_x = mapping_x[zero_tensor]
    transformed_3d_y = mapping_y[zero_tensor]
    transformed_3d_z = mapping_z[zero_tensor]

    transformed_2d_x = transformed_3d_x.view(zero_tensor.size(0),-1)
    transformed_2d_y = transformed_3d_y.view(zero_tensor.size(0),-1)
    transformed_2d_z = transformed_3d_z.view(zero_tensor.size(0),-1)

    triple_zeros = torch.repeat_interleave(zero_tensor, repeats=3, dim=1)

    zeros_xyz = torch.stack([transformed_2d_x, transformed_2d_y, transformed_2d_z], dim=0)

    total_atoms = atoms.get_number_of_atoms()

    # 9) Return with the same keys
    return {
        "triple_zeros"      : triple_zeros,
        "zeros"             : zero_tensor,
        "grain_types"       : grain_types_out,                 # shape (total_grains,)
        "center_of_mass"    : center_of_mass,                  # (total_grains,3)
        "total_force"       : total_force,                     # (total_grains,3)
        "relative_forces"   : final_relative_forces,           # (total_grains, 3*total_count)
        "sum_forces"        : final_forces,
        "relative_positions": final_relative_positions,        # (total_grains, 3*total_count)
        "grain_atomic_ids"  : final_grain_atomic_ids,          # (total_grains, total_count)
        "offsets"           : offsets_return,
        "grain_atom_counts" : grain_atom_counts,
        "total_atoms"       : total_atoms,
        "zeros_xyz"          : zeros_xyz,
        "zeros_x"          : transformed_2d_x,
        "zeros_y"          : transformed_2d_y,
        "zeros_z"          : transformed_2d_z,
    }


def calculate_grain_properties_unlabeled(atoms, grain_atom_counts):
    """
    Builds 'grains' based on grain_type and grain_num.
    Each grain_num groups certain grains together.
    Each type t has grain_atom_counts[t] atoms per grain.
    
    Returns the same keys as the old function:
      "grain_types", "center_of_mass", "average_force",
      "relative_positions", "relative_forces", "grain_atomic_ids",
      "offsets", "grain_atom_counts"
    """

    t1 = time.time()

    # 1) Basic retrieval
    grain_types_arr = atoms.get_array("grain_type").astype(int)  # (N_atoms,)
    grain_nums_arr  = atoms.get_array("grain_num").astype(int)    # (N_atoms,)
    positions = atoms.get_positions()                            # (N_atoms,3)
    cell      = atoms.get_cell()                                 # (3,3)
    inv_cell  = np.linalg.inv(cell)
    pbc       = atoms.get_pbc()                                  # (bool,bool,bool)

    # 2) Group atoms by (grain_type, grain_num)
    type_num_dict = defaultdict(list)
    N = len(grain_types_arr)
    for i in range(N):
        t = grain_types_arr[i]
        num = grain_nums_arr[i]
        key = (t, num)
        type_num_dict[key].append(i)

    unique_type_num = sorted(type_num_dict.keys())

    # group atoms by type
    type_dict = defaultdict(list)
    N = len(grain_types_arr)
    for i in range(N):
        t = grain_types_arr[i]
        num = grain_nums_arr[i]
        key = (t,num)
        type_dict[t].append(i)

    # 3) Build column offsets (as in old code)
    unique_types = sorted(grain_atom_counts.keys())
    type_num_grains = {}
    for t in unique_types:
        num_t = grain_atom_counts[t]
        n_atoms = len(type_dict[t])
        # number of grains = ceil(n_atoms / num_t)
        G_t = (n_atoms + num_t - 1) // num_t
        type_num_grains[t] = G_t

    total_count = sum(grain_atom_counts.values())

    # 4) Calculate how many "grains" (chunks) per (type, num)
    type_num_num_grains = {}
    for key in unique_type_num:
        t, num = key
        num_t = grain_atom_counts[t]
        n_atoms = len(type_num_dict[key])
        G_t = (n_atoms + num_t - 1) // num_t  # ceiling division
        type_num_num_grains[key] = G_t

    # total grains = sum of G_t
    total_grains = sum(type_num_num_grains[key] for key in unique_type_num)

    # 5) Prepare final arrays
    final_relative_positions = np.zeros((total_grains, 3*total_count), dtype=float)
    final_grain_atomic_ids   = -1 * np.ones((total_grains, total_count), dtype=int)
    center_of_mass = np.zeros((total_grains, 3), dtype=float)
    grain_types_out = np.zeros(total_grains, dtype=int)

    # 6) Row offset per (type, num)
    offsets = {}
    offsets_return = []
    running_row = 0
    for key in sorted(grain_atom_counts.keys()):
        offsets[key] = running_row
        start = running_row
        running_row += grain_atom_counts[key]
        offsets_return.append((start,running_row))


    t_middle = time.time()

    #print ("middle",t_middle-t1)

    t_pbc = 0
    t_positions_t = 0
    t_sort = 0
    t_correct_2d = 0
    t_reshape = 0
    t_fill = 0

    # 7) For each (type, num), zero-pad & reshape => (G_t, num_t, 3)
    for i,key in enumerate(unique_type_num):
        t, num = key
        atoms_for_t = type_num_dict[key]

        dt1 = time.time()
        atoms_for_t.sort()  # optional
        dt2 = time.time()

        t_sort += dt2-dt1

        num_t = grain_atom_counts[t]

        row0 = offsets[t]
        row1 = row0 + num_t

        # Fill grain_types_out
        grain_types_out[i] = t

        dt1 = time.time()
        # positions_t, forces_t => shape (n_atoms, 3)
        positions_t = positions[atoms_for_t]
        dt2 = time.time()
        
        t_positions_t += dt2-dt1

        dt1 = time.time()
        corrected_2d = find_closest_pbc_image_3d(positions_t, cell, inv_cell, pbc)
        dt2 = time.time()

        t_pbc += dt2-dt1
      
        # center_of_mass, average_force, total_force => shape (G_t,3)
        dt1 = time.time()
        cm = corrected_2d.mean(axis=0)
        dt2 = time.time()

        t_correct_2d += dt2-dt1

        center_of_mass [i] = cm

        # relative
        rel_pos_2d = corrected_2d - cm

        dt1 = time.time()
        position_final = rel_pos_2d.reshape(3*num_t)
        dt2 = time.time()
        t_reshape += dt2-dt1

        # fill final arrays

        dt1 = time.time()
        col0 = 3 * offsets[t]
        col1 = col0 + 3*num_t


        final_relative_positions[i, col0:col1] = position_final

        # fill grain_atomic_ids
        idx_array = -1 * np.ones((total_count,), dtype=int)
        idx_array[offsets[t]:offsets[t]+num_t] = np.array(atoms_for_t,dtype=int)
        final_grain_atomic_ids[i] = idx_array

        dt2 = time.time()
        t_fill += dt2-dt1

    t2 = time.time()

    #print ("sort",t_sort)
    #print ("pbc",t_pbc)
    #print ("positions_t",t_positions_t)
    #print ("correct_t", t_correct_2d)
    #print ("reshape",t_reshape)
    #print ("fill",t_fill)
    zero_tensor = torch.zeros((len(grain_types_out), sum(grain_atom_counts.values())), dtype=torch.long)
    start_idx = 0
    for grain_type, count in grain_atom_counts.items():

        end_idx = start_idx + count

        mask = torch.tensor(grain_types_out == grain_type) 
        mask_indices = torch.nonzero(mask, as_tuple=True)[0]

        # Fill zero_tensor
        zero_tensor[mask_indices, start_idx:end_idx] = 1.0

        start_idx = end_idx

    mapping_x = torch.tensor([[0,0,0],[1,0,0]], dtype=torch.long)
    mapping_y = torch.tensor([[0,0,0],[0,1,0]], dtype=torch.long)
    mapping_z = torch.tensor([[0,0,0],[0,0,1]], dtype=torch.long)
   
    transformed_3d_x = mapping_x[zero_tensor]
    transformed_3d_y = mapping_y[zero_tensor]
    transformed_3d_z = mapping_z[zero_tensor]

    transformed_2d_x = transformed_3d_x.view(zero_tensor.size(0),-1)
    transformed_2d_y = transformed_3d_y.view(zero_tensor.size(0),-1)
    transformed_2d_z = transformed_3d_z.view(zero_tensor.size(0),-1)

    triple_zeros = torch.repeat_interleave(zero_tensor, repeats=3, dim=1)

    zeros_xyz = torch.stack([transformed_2d_x, transformed_2d_y, transformed_2d_z], dim=0)

    total_atoms = atoms.get_number_of_atoms()

    # 9) Return with the same keys
    return {
        "triple_zeros"      : triple_zeros,
        "zeros"             : zero_tensor,
        "grain_types"       : grain_types_out,                 # shape (total_grains,)
        "center_of_mass"    : center_of_mass,                  # (total_grains,3)
        "relative_positions": final_relative_positions,        # (total_grains, 3*total_count)
        "grain_atomic_ids"  : final_grain_atomic_ids,          # (total_grains, total_count)
        "offsets"           : offsets_return,
        "grain_atom_counts" : grain_atom_counts,
        "total_atoms"       : total_atoms,
        "zeros_xyz"          : zeros_xyz,
        "zeros_x"          : transformed_2d_x,
        "zeros_y"          : transformed_2d_y,
        "zeros_z"          : transformed_2d_z,
    }

def _graph_build_matscipy(cutoff: float, pbc, cell, pos):
    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]

    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(pos)) + 1

    # Extend cell in non-periodic directions
    # For models with more than 5 layers,
    # the multiplicative constant needs to be increased.
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]
    # it does not have self-interaction
    edge_src, edge_dst, edge_vec, shifts = neighbour_list(
        quantities='ijDS',
        pbc=pbc,
        cell=cell,
        positions=pos,
        cutoff=cutoff,
    )
    # dtype issue
    edge_src = edge_src.astype(np.int64)
    edge_dst = edge_dst.astype(np.int64)

    return edge_src, edge_dst, edge_vec, shifts



def _graph_build_ase(cutoff: float, pbc, cell, pos):
    # building neighbor list
    edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
        'ijDS', pbc, cell, pos, cutoff, self_interaction=True
    )

    is_zero_idx = np.all(edge_vec == 0, axis=1)
    is_self_idx = edge_src == edge_dst
    non_trivials = ~(is_zero_idx & is_self_idx)
    shifts = np.array(shifts[non_trivials])

    edge_vec = edge_vec[non_trivials]
    edge_src = edge_src[non_trivials]
    edge_dst = edge_dst[non_trivials]


    return edge_src, edge_dst, edge_vec, shifts


_graph_build_f = _graph_build_ase
try:
    from matscipy.neighbours import neighbour_list

    _graph_build_f = _graph_build_matscipy
except ImportError:
    pass


def _correct_scalar(v):
    if isinstance(v, np.ndarray):
        v = v.squeeze()
        assert v.ndim == 0, f'given {v} is not a scalar'
        return v
    elif isinstance(v, (int, float, np.integer, np.floating)):
        return np.array(v)
    else:
        assert False, f'{type(v)} is not expected'


def unlabeled_atoms_to_graph(atoms: ase.Atoms, cutoff: float, grain_atom_counts: dict):

    t1 = time.time()
    grain_info = calculate_grain_properties_unlabeled(atoms,grain_atom_counts) 
    t2 = time.time()
    #print("grain info",t2-t1)

    pos = grain_info['center_of_mass']
    relative_pos = grain_info['relative_positions']
    cell = np.array(atoms.get_cell())
    pbc = atoms.get_pbc()

    t1 = time.time()
    edge_src, edge_dst, edge_vec, shifts = _graph_build_f(cutoff, pbc, cell, pos)
    t2 = time.time()

    #print("edge",t2-t1)

    edge_idx = np.array([edge_src, edge_dst])

    atomic_numbers = grain_info['grain_types'] 

    total_atoms    = grain_info['total_atoms']

    zeros = grain_info["zeros"]
    triple_zeros = grain_info["triple_zeros"]

    cell = np.array(cell)
    vol = _correct_scalar(atoms.cell.volume)
    if vol == 0:
        vol = np.array(np.finfo(float).eps)

    zeros = grain_info["zeros"]
    triple_zeros = grain_info["triple_zeros"]

    tot_num = _correct_scalar(len(atomic_numbers))
    force_save = torch.zeros(tot_num,3,dtype=float)

    zeros_xyz = grain_info['zeros_xyz']
    zeros_x = grain_info['zeros_x']
    zeros_y = grain_info['zeros_y']
    zeros_z = grain_info['zeros_z']


    data = {
        KEY.ZERO_ARRAY:   zeros,
        KEY.TRIPLE_ZERO:  triple_zeros,
        KEY.TOTAL_ATOMS:  total_atoms, 
        KEY.NODE_FEATURE: atomic_numbers,
        KEY.ATOMIC_NUMBERS: atomic_numbers,
        KEY.POS: pos,
        KEY.RELATIVE_POS: relative_pos,
        KEY.EDGE_IDX: edge_idx,
        KEY.EDGE_VEC: edge_vec,
        KEY.CELL: cell,
        KEY.CELL_SHIFT: shifts,
        KEY.CELL_VOLUME: vol,
        KEY.NUM_ATOMS: _correct_scalar(len(atomic_numbers)),
        KEY.ATOM_IDS: grain_info['grain_atomic_ids'],
        KEY.OFFSETS:  grain_info['offsets'],
        KEY.FORCE_SAVE: force_save,
        KEY.ZEROS_X:  zeros_x,
        KEY.ZEROS_Y:  zeros_y,
        KEY.ZEROS_Z:  zeros_z,
    }
    data[KEY.INFO] = {}
    return data


def atoms_to_graph(
    atoms: ase.Atoms,
    cutoff: float,
    grain_atom_counts = dict,
    transfer_info: bool = True,
    y_from_calc: bool = False,
):
    """
    From ase atoms, return AtomGraphData as graph based on cutoff radius
    Except for energy, force and stress labels must be numpy array type
    as other cases are not tested.
    Returns 'np.nan' with consistent shape for unlabeled data
    (ex. stress of non-pbc system)

    Args:
        atoms (Atoms): ase atoms
        cutoff (float): cutoff radius
        transfer_info (bool): if True, transfer ".info" from atoms to graph,
                              defaults to True
        y_from_calc: if True, get ref values from calculator, defaults to False
    Returns:
        numpy dict that can be used to initialize AtomGraphData
        by AtomGraphData(**atoms_to_graph(atoms, cutoff))
        , for scalar, its shape is (), and types are np.ndarray
    Requires grad is handled by 'dataset' not here.
    """

    # grain info part
    grain_info = calculate_grain_properties(atoms,grain_atom_counts) 


    if not y_from_calc:
        y_energy = atoms.info['y_energy']
        #y_force = atoms.arrays['y_force']
        y_stress = atoms.info.get('y_stress', np.full((6,), np.nan))
        if y_stress.shape == (3, 3):
            y_stress = np.array(
                [
                    y_stress[0][0],
                    y_stress[1][1],
                    y_stress[2][2],
                    y_stress[0][1],
                    y_stress[1][2],
                    y_stress[2][0],
                ]
            )
        else:
            y_stress = y_stress.squeeze()
    else:
        try:
            y_energy = atoms.get_potential_energy(force_consistent=True)
        except NotImplementedError:
            y_energy = atoms.get_potential_energy()
#        y_force = atoms.get_forces(apply_constraint=False)
        try:
            y_stress = -1 * atoms.get_stress()  # it ensures correct shape
            y_stress = np.array(y_stress[[0, 1, 2, 5, 3, 4]])
        except RuntimeError:
            y_stress = np.full((6,), np.nan)
    assert y_stress.shape == (6,), 'If you see this, please report to the maintainer'

    y_force = grain_info['total_force']
    y_relative_force = grain_info['relative_forces']
    sum_forces = grain_info['sum_forces'] 


    pos = grain_info['center_of_mass']
    relative_pos = grain_info['relative_positions']

    cell = np.array(atoms.get_cell())
    pbc = atoms.get_pbc()


    edge_src, edge_dst, edge_vec, shifts = _graph_build_f(cutoff, pbc, cell, pos)

    edge_idx = np.array([edge_src, edge_dst])
    atomic_numbers = grain_info['grain_types'] 

    cell = np.array(cell)
    vol = _correct_scalar(atoms.cell.volume)
    if vol == 0:
        vol = np.array(np.finfo(float).eps)

    total_atoms = grain_info["total_atoms"]

    zeros = grain_info["zeros"]
    triple_zeros = grain_info["triple_zeros"]

    tot_num = _correct_scalar(len(atomic_numbers))
    force_save = torch.zeros(tot_num,3,dtype=float)

    zeros_xyz = grain_info['zeros_xyz']
    zeros_x = grain_info['zeros_x']
    zeros_y = grain_info['zeros_y']
    zeros_z = grain_info['zeros_z']

    data = {
        KEY.ZERO_ARRAY:   zeros,
        KEY.TRIPLE_ZERO:  triple_zeros,
        KEY.TOTAL_ATOMS:  total_atoms, 
        KEY.NODE_FEATURE: atomic_numbers,
        KEY.ATOMIC_NUMBERS: atomic_numbers,
        KEY.POS: pos,
        KEY.RELATIVE_POS: relative_pos,
        KEY.EDGE_IDX: edge_idx,
        KEY.EDGE_VEC: edge_vec,
        KEY.ENERGY: _correct_scalar(y_energy),
        KEY.FORCE: y_force,
        KEY.RELATIVE_FORCE: y_relative_force,
        KEY.SUM_FORCE: sum_forces,
        KEY.STRESS: y_stress.reshape(1, 6),  # to make batch have (n_node, 6)
        KEY.CELL: cell,
        KEY.CELL_SHIFT: shifts,
        KEY.CELL_VOLUME: vol,
        KEY.NUM_ATOMS: _correct_scalar(len(atomic_numbers)),
        KEY.PER_ATOM_ENERGY: _correct_scalar(y_energy / len(pos)),
        KEY.ATOM_IDS: grain_info['grain_atomic_ids'],
        KEY.OFFSETS:  grain_info['offsets'],
        KEY.FORCE_SAVE: force_save,
        KEY.ZEROS_X:  zeros_x,
        KEY.ZEROS_Y:  zeros_y,
        KEY.ZEROS_Z:  zeros_z,
    }

    if transfer_info and atoms.info is not None:
        info = copy.deepcopy(atoms.info)
        # save only metadata
        # TODO: is it really necessary?
        if 'y_energy' in info:
            del info['y_energy']
        if 'y_force' in info:
            del info['y_force']
        if 'y_stress' in info:
            del info['y_stress']
        data[KEY.INFO] = info

    else:
        data[KEY.INFO] = {}

    return data


def graph_build(
    atoms_list: List,
    cutoff: float,
    grain_atom_counts: dict,
    num_cores: int = 1,
    transfer_info: bool = True,
    y_from_calc: bool = False,
) -> List[AtomGraphData]:
    """
    parallel version of graph_build
    build graph from atoms_list and return list of AtomGraphData
    Args:
        atoms_list (List): list of ASE atoms
        cutoff (float): cutoff radius of graph
        num_cores (int): number of cores to use
        transfer_info (bool): if True, copy info from atoms to graph,
                              defaults to True
        y_from_calc (bool): Get reference y labels from calculator, defaults to False
    Returns:
        List[AtomGraphData]: list of AtomGraphData
    """
    serial = num_cores == 1
    inputs = [(atoms, cutoff, grain_atom_counts, transfer_info, y_from_calc) for atoms in atoms_list]


    if not serial:
        pool = mp.Pool(num_cores)
        graph_list = pool.starmap(
            atoms_to_graph,
            tqdm(inputs, total=len(atoms_list), desc=f'graph_build ({num_cores})'),
        )
        pool.close()
        pool.join()
    else:
        graph_list = [
            atoms_to_graph(*input_)
            for input_ in tqdm(inputs, desc='graph_build (1)')
        ]

    
    graph_list = [AtomGraphData.from_numpy_dict(g) for g in graph_list]
    
    return graph_list


def _set_atoms_y(
    atoms_list: list[ase.Atoms],
    energy_key: Optional[str] = None,
    force_key: Optional[str] = None,
    stress_key: Optional[str] = None,
) -> list[ase.Atoms]:
    """
    Define how SevenNet reads ASE.atoms object for its y label
    If energy_key, force_key, or stress_key is given, the corresponding
    label is obtained from .info dict of Atoms object. These values should
    have eV, eV/Angstrom, and eV/Angstrom^3 for energy, force, and stress,
    respectively. (stress in Voigt notation)

    Args:
        atoms_list (list[ase.Atoms]): target atoms to set y_labels
        energy_key (str, optional): key to get energy. Defaults to None.
        force_key (str, optional): key to get force. Defaults to None.
        stress_key (str, optional): key to get stress. Defaults to None.

    Returns:
        list[ase.Atoms]: list of ase.Atoms

    Raises:
        RuntimeError: if ase atoms are somewhat imperfect

    Use free_energy: atoms.get_potential_energy(force_consistent=True)
    If it is not available, use atoms.get_potential_energy()
    If stress is available, initialize stress tensor
    Ignore constraints like selective dynamics
    """
    for atoms in atoms_list:
        # access energy
        if energy_key is not None:
            atoms.info['y_energy'] = atoms.info[energy_key]
            del atoms.info[energy_key]
        else:
            try:
                atoms.info['y_energy'] = atoms.get_potential_energy(
                    force_consistent=True
                )
            except NotImplementedError:
                atoms.info['y_energy'] = atoms.get_potential_energy()
        # access force
        if force_key is not None:
            atoms.arrays['y_force'] = atoms.arrays[force_key]
            del atoms.arrays[force_key]
        else:
            atoms.arrays['y_force'] = atoms.get_forces(apply_constraint=False)
        # access stress
        if stress_key is not None:
            y_stress = -1 * atoms.info[stress_key]
            atoms.info['y_stress'] = np.array(y_stress[[0, 1, 2, 5, 3, 4]])
            del atoms.info[stress_key]
        else:
            try:
                # xx yy zz xy yz zx order
                # We expect this is eV/A^3 unit
                # (ASE automatically converts vasp kB to eV/A^3)
                # So we restore it
                y_stress = -1 * atoms.get_stress()
                atoms.info['y_stress'] = np.array(y_stress[[0, 1, 2, 5, 3, 4]])
            except RuntimeError:
                atoms.info['y_stress'] = np.full((6,), np.nan)
    return atoms_list


def ase_reader(
    filename: str,
    energy_key: Optional[str] = None,
    force_key: Optional[str] = None,
    stress_key: Optional[str] = None,
    index: str = ':',
    **kwargs,
) -> list[ase.Atoms]:
    """
    Wrapper of ase.io.read
    """
    atoms_list = ase.io.read(filename, index=index, **kwargs)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    return _set_atoms_y(atoms_list, energy_key, force_key, stress_key)


# Reader
def structure_list_reader(filename: str, format_outputs='vasp-out'):
    """
    Deprecated
    Read from structure_list using braceexpand and ASE

    Args:
        fname : filename of structure_list

    Returns:
        dictionary of lists of ASE structures.
        key is title of training data (user-define)
    """
    parsers = DefaultParsersContainer(
        PositionsAndForces, Stress, Energy, Cell
    ).make_parsers()
    ocp = OutcarChunkParser(parsers=parsers)

    def parse_label(line):
        line = line.strip()
        if line.startswith('[') is False:
            return False
        elif line.endswith(']') is False:
            raise ValueError('wrong structure_list title format')
        return line[1:-1]

    def parse_fileline(line):
        line = line.strip().split()
        if len(line) == 1:
            line.append(':')
        elif len(line) != 2:
            raise ValueError('wrong structure_list format')
        return line[0], line[1]

    structure_list_file = open(filename, 'r')
    lines = structure_list_file.readlines()

    raw_str_dict = {}
    label = 'Default'
    for line in lines:
        if line.strip() == '':
            continue
        tmp_label = parse_label(line)
        if tmp_label:
            label = tmp_label
            raw_str_dict[label] = []
            continue
        elif label in raw_str_dict:
            files_expr, index_expr = parse_fileline(line)
            raw_str_dict[label].append((files_expr, index_expr))
        else:
            raise ValueError('wrong structure_list format')
    structure_list_file.close()

    structures_dict = {}
    info_dct = {'data_from': 'user_OUTCAR'}
    for title, file_lines in raw_str_dict.items():
        stct_lists = []
        for file_line in file_lines:
            files_expr, index_expr = file_line
            index = string2index(index_expr)
            for expanded_filename in list(braceexpand(files_expr)):
                f_stream = open(expanded_filename, 'r')
                # generator of all outcar ionic steps
                gen_all = outcarchunks(f_stream, ocp)
                try:  # TODO: index may not slice, it can be integer
                    it_atoms = islice(gen_all, index.start, index.stop, index.step)
                except ValueError:
                    # TODO: support
                    # negative index
                    raise ValueError('Negative index is not supported yet')

                info_dct_f = {
                    **info_dct,
                    'file': os.path.abspath(expanded_filename),
                }
                for idx, o in enumerate(it_atoms):
                    try:
                        istep = index.start + idx * index.step
                        atoms = o.build()
                        atoms.info = {**info_dct_f, 'ionic_step': istep}
                    except TypeError:  # it is not slice of ionic steps
                        atoms = o.build()
                        atoms.info = info_dct_f
                    stct_lists.append(atoms)
                f_stream.close()
        structures_dict[title] = stct_lists
    return {k: _set_atoms_y(v) for k, v in structures_dict.items()}


def match_reader(reader_name: str, **kwargs):
    reader = None
    metadata = {}
    if reader_name == 'structure_list':
        reader = partial(structure_list_reader, **kwargs)
        metadata.update({'origin': 'structure_list'})
    else:
        reader = partial(ase_reader, **kwargs)
        metadata.update({'origin': 'ase_reader'})
    return reader, metadata


def file_to_dataset(
    file: str,
    cutoff: float,
    grain_atom_counts: dict,
    cores: int = 1,
    reader: Callable = ase_reader,
    label: Optional[str] = None,
    transfer_info: bool = True,
):
    """
    Deprecated
    Read file by reader > get list of atoms or dict of atoms
    """

    # expect label: atoms_list dct or atoms or list of atoms
    atoms = reader(file)

    if type(atoms) is list:
        if label is None:
            label = KEY.LABEL_NONE
        atoms_dct = {label: atoms}
    elif isinstance(atoms, ase.Atoms):
        if label is None:
            label = KEY.LABEL_NONE
        atoms_dct = {label: [atoms]}
    elif isinstance(atoms, dict):
        atoms_dct = atoms
    else:
        raise TypeError('The return of reader is not list or dict')

    graph_dct = {}
    for label, atoms_list in atoms_dct.items():
        graph_list = graph_build(
            atoms_list=atoms_list,
            cutoff=cutoff,
            grain_atom_counts = grain_atom_counts,
            num_cores=cores,
            transfer_info=transfer_info,
            y_from_calc=False,
        )
        for graph in graph_list:
            graph[KEY.USER_LABEL] = label
        graph_dct[label] = graph_list
    db = AtomGraphDataset(graph_dct, cutoff)
    return db
