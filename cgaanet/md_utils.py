import torch
import ase
from ase.units import fs

def set_velocity_mass(data,atoms):

    shape1,shape2 = data['grain_atomic_ids'].shape
    grain_atomic_ids = data['grain_atomic_ids'].reshape(-1)  

    vel0  = atoms.get_velocities() * fs
    mass0 = atoms.get_masses()

    vel  = torch.from_numpy(vel0).float().to(grain_atomic_ids.device)
    mass = torch.from_numpy(mass0).float().to(grain_atomic_ids.device)

    n = grain_atomic_ids.shape[0]

    v = torch.zeros(n, 3, device=grain_atomic_ids.device)
    m = torch.full((n,), 0.0001, device=grain_atomic_ids.device)

    valid_mask = grain_atomic_ids >= 0
    valid_ids  = grain_atomic_ids[valid_mask].long()

    v[valid_mask] = vel[valid_ids]
    m[valid_mask] = mass[valid_ids]


    data['velocity'] = v.reshape(shape1,3*shape2) / fs 
    data['mass']   = m.reshape(shape1,shape2)

    data['triple_mass'] = data['mass'].detach().clone().repeat_interleave(3,dim=1)

    return data

def positions_fullstep(data,modified_acc,dt,dtdt):

    data['relative_positions'] += dt * data['velocity'] + 0.5 * dtdt * modified_acc

    pos_x = data['relative_positions']*data['zeros_xyz'][0]
    pos_y = data['relative_positions']*data['zeros_xyz'][1]
    pos_z = data['relative_positions']*data['zeros_xyz'][2]

    ave_x = pos_x.sum(dim=1) / data['zeros'].sum(dim=1)
    ave_y = pos_y.sum(dim=1) / data['zeros'].sum(dim=1)
    ave_z = pos_z.sum(dim=1) / data['zeros'].sum(dim=1)

    ave_x = ave_x.unsqueeze(-1)
    ave_y = ave_y.unsqueeze(-1)
    ave_z = ave_z.unsqueeze(-1)


    d_relative_pos = ave_x*data['zeros_xyz'][0] + ave_y*data['zeros_xyz'][1] + ave_z*data['zeros_xyz'][2]
    d_pos = torch.cat([ave_x,ave_y,ave_z],dim=1)

    data['relative_positions'] -= d_relative_pos 
    data['pos'] += d_pos

    return data
