"""Stage this node's data"""
import argparse
import os
from mpi4py import MPI

import distutils
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("-n", "--n-files", type=int,
                        help="Total number of files to be staged")
    parser.add_argument("-u", "--update", action="store_true",
                        help="Update new files only")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Turn on verbose logging")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    files = sorted(os.listdir(args.input_dir))
    if args.n_files is not None:
        files = files[:args.n_files]
    #files = files[:world_size]

    for f in files[rank::world_size]:
        if args.verbose:
            print('Copying', f)
        shutil.copyfile(
            os.path.join(args.input_dir, f),
            os.path.join(args.output_dir, f),
        )
