"""Utility code for staging data files into local storage"""

# System imports
import os
import shutil
import logging

def stage_files(input_dir, output_dir, n_files, rank=0, size=1):
    """Stage specified number of files to directory.

    This function works in a distributed fashion. Each rank will only stage
    its chunk of the file list.
    """
    if rank == 0:
        logging.info(f'Staging {n_files} files to {output_dir}')

    # Find all the files in the input directory
    files = sorted(os.listdir(input_dir))

    # Make sure there are at least enough files available
    if len(files) < n_files:
        raise ValueError(f'Cannot stage {n_files} files; only {len(files)} available')

    # Take the specified number of files
    files = files[:n_files]

    # Copy my chunk into the output directory
    os.makedirs(output_dir, exist_ok=True)
    for f in files[rank::size]:
        logging.debug(f'Staging file {f}')
        shutil.copyfile(os.path.join(input_dir, f),
                        os.path.join(output_dir, f))
    logging.debug('Data staging completed')
