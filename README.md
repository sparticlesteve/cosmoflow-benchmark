# CosmoFlow TensorFlow Keras benchmark implementation

This is a an implementation of the
[CosmoFlow](https://arxiv.org/abs/1808.04728) 3D convolutional neural network
for benchmarking. It is written in TensorFlow with the Keras API and uses
[Horovod](https://github.com/horovod/horovod) for distributed training.

## Datasets

Globus is the current recommended way to transfer the dataset locally.
There is a globus endpoint at:

https://app.globus.org/file-manager?origin_id=d0b1b73a-efd3-11e9-993f-0a8c187e8c12&origin_path=%2F

The latest pre-processed dataset in TFRecord format is in the
`cosmoUniverse_2019_05_4parE_tf` folder, which contains training and validation
subfolders. There are currently 262144 samples for training and 65536 samples
for validation/testing.

For the previous dataset which was used for the 2020 ECP Annual Meeting results,
you can use the `cosmoUniverse_2019_02_4parE_dim128_cube_nT4.tar` tarball.
This is a 2.2 TB tar file containing 1027 `TFRecord` files, each representing
a simulated universe with 64 sub-volume samples.

## Running the benchmark

Submission scripts are in `scripts`. YAML configuration files go in `configs`.

### Running at NERSC

`sbatch -N 64 scripts/train_cori.sh`
