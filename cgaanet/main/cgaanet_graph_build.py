import argparse
import glob
import os
import sys
from datetime import datetime

from cgaanet import __version__

description = (
    f'cgaanet version={__version__}, cgaanet_graph_build.\n'
    + 'Create `cgaanet_data/dataset.pt` or `.cgaanet_data` from '
    + 'ase readable or VASP OUTCARs (by structure_list).\n'
)

source_help = 'source data to build graph, knows *'
cutoff_help = 'cutoff radius of edges in Angstrom'
filename_help = (
    'Name of the dataset, default is graph.pt. '
    + 'The dataset will be written under "cgaanet_data", '
    + 'for example, {out}/cgaanet_data/graph.pt.'
)
legacy_help = 'build legacy .cgaanet_data'


def run(args):
    import cgaanet.scripts.graph_build as graph_build
    from cgaanet.cgaanet_logger import Logger

    source = glob.glob(args.source)
    cutoff = args.cutoff
    num_cores = args.num_cores
    filename = args.filename
    out = args.out
    legacy = args.legacy
    fmt_kwargs = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            k, v = kwarg.split('=')
            fmt_kwargs[k] = v

    if len(source) == 0:
        print('Source has zero len, nothing to read')
        sys.exit(0)

    if not os.path.isdir(out):
        raise NotADirectoryError(f'No such directory: {out}')

    to_be_written = os.path.join(out, 'cgaanet_data', filename)
    if os.path.isfile(to_be_written):
        raise FileExistsError(f'File already exist: {to_be_written}')

    metadata = {
        'cgaanet_version': __version__,
        'when': datetime.now().strftime('%Y-%m-%d'),
        'cutoff': cutoff,
    }

    with Logger(filename=None, screen=args.screen) as logger:
        logger.writeline(description)

        if not legacy:
            graph_build.build_cgaanet_graph_dataset(
                source,
                cutoff,
                num_cores,
                out,
                filename,
                metadata,
                **fmt_kwargs,
            )
        else:
            out = os.path.join(out, filename.split('.')[0])
            graph_build.build_script(  # build .cgaanet_data
                source,
                cutoff,
                num_cores,
                out,
                metadata,
                **fmt_kwargs,
            )


def main():
    ag = argparse.ArgumentParser(description=description)

    ag.add_argument('source', help=source_help, type=str)
    ag.add_argument('cutoff', help=cutoff_help, type=float)
    ag.add_argument(
        '-n',
        '--num_cores',
        help='number of cores to build graph in parallel',
        default=1,
        type=int,
    )
    ag.add_argument(
        '-o',
        '--out',
        help='Existing path to write outputs.',
        type=str,
        default='./',
    )
    ag.add_argument(
        '-f',
        '--filename',
        help=filename_help,
        type=str,
        default='graph.pt',
    )
    ag.add_argument(
        '--legacy',
        help=legacy_help,
        action='store_true',
    )
    ag.add_argument(
        '-s',
        '--screen',
        help='print log to the screen',
        action='store_true',
    )
    ag.add_argument(
        '--kwargs',
        nargs=argparse.REMAINDER,
        help='will be passed to ase.io.read, or can be used to specify EFS key',
    )

    args = ag.parse_args()
    run(args)
