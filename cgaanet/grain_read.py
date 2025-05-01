from collections import defaultdict
import cgaanet._keys as KEY
import torch

def preprocess_grain_info(config):
    """
    Extracts grain information from an AtomGraphDataset-like object and
    stores the resulting list of grain types, grain type map, and grain atom counts
    into the given config dictionary.

    Args:
        graphs: An AtomGraphDataset or a similar object. Must support `to_list()`,
                which returns a list of data samples (e.g., AtomGraphData).
        config: A dictionary where the resulting grain info will be stored.

    The following keys are updated in config:
        - "GRAIN_TYPES": A sorted list of unique grain types.
        - "GRAIN_TYPE_MAP": A dictionary mapping each grain type to an index.
        - "GRAIN_ATOM_COUNTS": A dictionary with grain type as key and total atom count as value.
    """

    grain_atom_counts = config[KEY.GRAIN_ATOM_COUNTS]


    sorted_grain_types = sorted(grain_atom_counts.keys()) 

    # 6. Update the config
    config.update({
        "GRAIN_TYPES": sorted_grain_types,     # list
        "GRAIN_ATOM_COUNTS": grain_atom_counts # dict
    })

