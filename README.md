# CGAANet

This is the prototype version of CGAANet. 

Note that this code is not yet at a stage suitable for practical use, but is rather a prototype created to demonstrate the concept of CGAA-FF.

Therefore, only training and ASE simulator are available, and other features, such as multi-GPU and LAMMPS implementaion, are not supported for now.

## Training

This code is developed by modifying SevenNet code (https://github.com/MDIL-SNU/SevenNet/).

Therefore, input formats are similar to SevenNet.

### Command

Only train_v1 is supported. 

```python
cgaanet -m train_v1 input.yaml -s
```

### input details

You have to add below keys, which are not included in the SeveNet code:

```python
model:
    grain_atom_counts: {1: 1, 2: 7, 3: 10, 4: 15}
    error_record:
        - ['IntraGrainForce', 'RMSE']
```

Here, grain_atom_counts is the number of atoms in grains.


## Preparation of the training set
Example:
```python
2
Lattice="110.0 0.0 0.0 1.78e-13 110.0 0.0 1.78e-13 1.78e-13 110.0" Properties=species:S:1:pos:R:3:momenta:R:3:grain_num:I:1:grain_type:I:1:intra_grain_sequence:I:1:forces:R:3 energy=1.0 pbc="T T T"
O       2.16470000        1.10030000      1.78530000      -0.18473615      -0.01513482      -0.12696557        0        1        1       0.51619900      -0.00537521      -0.03544830
S       108.16470000      2.20030000      5.78530000      -0.18473615      -0.01513482      -0.12696557        0        1        0       0.11619900      -0.00537521      -0.03544830
```
You have to include grain_num, grain_type, and intra_grain_sequence tag in the extxyz format.

- grain_num: index for grain (start from 0)
- grain_type: index for grain types (should be start from 1)
- intra_grain_sequence: predefined order for grain embeddings


## ASE Simulator 

```python
from cgaanet.cgaanet_calculator import CGAANetCalculator
calc = CGAANetCalculator(model='checkpoint_best.pth', device='cuda')
```

## Citation

