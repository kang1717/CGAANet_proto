import os
from typing import Optional

from cgaanet.cgaanet_logger import Logger
from cgaanet.train.dataset import AtomGraphDataset
from cgaanet.util import unique_filepath


def build_cgaanet_graph_dataset(
    source: list[str],
    cutoff: float,
    num_cores: int,
    out: str,
    filename: str,
    metadata: Optional[dict] = None,
    **fmt_kwargs,
):
    from cgaanet.train.graph_dataset import SevenNetGraphDataset

    log = Logger()
    if metadata is None:
        metadata = {}

    log.timer_start('graph_build')
    db = SevenNetGraphDataset(
        cutoff=cutoff,
        root=out,
        files=source,
        processed_name=filename,
        process_num_cores=num_cores,
        **fmt_kwargs,
    )
    log.timer_end('graph_build', 'graph build time')
    log.writeline(f'Graph saved: {db.processed_paths[0]}')

    log.bar()
    for k, v in metadata.items():
        log.format_k_v(k, v, write=True)
    log.bar()

    log.writeline('Distribution:')
    log.statistic_write(db.statistics)
    log.format_k_v('# atoms (node)', db.natoms, write=True)
    log.format_k_v('# structures (graph)', len(db), write=True)


def dataset_finalize(dataset, metadata, out):
    """
    Deprecated
    """
    natoms = dataset.get_natoms()
    species = dataset.get_species()
    metadata = {
        **metadata,
        'natoms': natoms,
        'species': species,
    }
    dataset.meta = metadata

    if os.path.isdir(out):
        out = os.path.join(out, 'graph_built.cgaanet_data')
    elif out.endswith('.cgaanet_data') is False:
        out = out + '.cgaanet_data'
    out = unique_filepath(out)

    log = Logger()
    log.writeline('The metadata of the dataset is...')
    for k, v in metadata.items():
        log.format_k_v(k, v, write=True)
    dataset.save(out)
    log.writeline(f'dataset is saved to {out}')

    return dataset


def build_script(
    source: list[str],
    cutoff: float,
    grain_atom_counts: dict,
    num_cores: int,
    out: str,
    metadata: Optional[dict] = None,
    **fmt_kwargs,
):
    """
    Deprecated
    """
    from cgaanet.train.dataload import file_to_dataset, match_reader

    if metadata is None:
        metadata = {}
    log = Logger()

    dataset = AtomGraphDataset({}, cutoff)
    common_args = {
        'cutoff': cutoff,
        'cores': num_cores,
        'label': 'graph_build',
        'grain_atom_counts': grain_atom_counts,
    }
    log.timer_start('graph_build')
    for path in source:
        if os.path.isdir(path):
            continue
        log.writeline(f'Read: {path}')
        basename = os.path.basename(path)
        if 'structure_list' in basename:
            fmt = 'structure_list'
        else:
            fmt = 'ase'
        reader, rmeta = match_reader(fmt, **fmt_kwargs)
        metadata.update(**rmeta)
        dataset.augment(
            file_to_dataset(
                file=path,
                reader=reader,
                **common_args,
            )
        )
    log.timer_end('graph_build', 'graph build time')
    dataset_finalize(dataset, metadata, out)
