"""Stage this node's data"""
import argparse
import os
from mpi4py import MPI

import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Output directory",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    files = sorted(os.listdir(args.input_dir))
    files = files[:world_size]

    for f in files[rank::world_size]:
        shutil.copyfile(
            os.path.join(args.input_dir, f),
            os.path.join(args.output_dir, f),
        )
        
