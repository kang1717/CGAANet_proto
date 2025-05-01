# CGAANet

This is the prototype version of CGAANet. 

Note that this code is not yet at a stage suitable for practical use, but is rather a prototype created to demonstrate the concept of CGAA-FF.

Therefore, only training and ASE simulator are available, and other features, such as multi-GPU and LAMMPS implementaion, are not supported for now.

## Training

This code is developed by modifying SevenNet code (https://github.com/MDIL-SNU/SevenNet/).

### Command
```python
cgaanet -m train_v1 input.yaml -s
```

### input details (difference between SevenNet)



## ASE Simulator 

```python
from cgaanet.cgaanet_calculator import CGAANetCalculator
calc = CGAANetCalculator(model='checkpoint_best.pth', device='cuda')
```
