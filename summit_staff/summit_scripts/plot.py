import os
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def load_config(result_dir):
    config_file = os.path.join(result_dir, 'config.pkl')
    with open(config_file, 'rb') as f:
        return pickle.load(f)

def load_result(result_dir):
    history_file = os.path.join(result_dir, 'history.csv')
    return pd.read_csv(history_file)

def compute_mean_time(r):
    return r[r.epoch>0].time.mean()

def get_num_samples(config):
    dc = config['data']
    return dc['n_train'] + dc['n_valid']

plt.rcParams.update({'font.size': 14})

#########################
## 1 node -- Strong Scale
ranks = np.array([1, 2, 3, 4, 5, 6])
results_pattern = os.path.expandvars('../summit_output/strong_scaling/log_%i')

configs, results = [], []
for r in ranks:
    result_dir = results_pattern % r
    configs.append(load_config(result_dir))
    results.append(load_result(result_dir).assign(ranks=r))

samples = np.array([get_num_samples(c) for (c,r) in zip(configs, ranks)])
times = np.array([compute_mean_time(r) for r in results])
throughputs = samples / times

scaling_cgpu = pd.DataFrame(dict(ranks=ranks, samples=samples, 
                                 times=times, throughputs=throughputs))

plt.figure(figsize=(9,8))
plt.plot(scaling_cgpu.ranks, scaling_cgpu.throughputs, 'o-', label='Real data')
plt.xlabel('Number of Summit-GPUs')
plt.ylabel('Training throughput [samples/s]')

plt.plot(scaling_cgpu.ranks, scaling_cgpu.ranks*scaling_cgpu.throughputs[0], '--', label='Ideal')
plt.legend(loc=0);
plt.show()

###########################
## 10 nodes -- Strong Scale
ranks = np.array([6, 12, 24, 32, 60])
results_pattern = os.path.expandvars('../summit_output/strong_scaling/log_%i')

configs, results = [], []
for r in ranks:
    result_dir = results_pattern % r
    configs.append(load_config(result_dir))
    results.append(load_result(result_dir).assign(ranks=r))

samples = np.array([get_num_samples(c) for (c,r) in zip(configs, ranks)])
times = np.array([compute_mean_time(r) for r in results])
throughputs = samples / times

scaling_cgpu = pd.DataFrame(dict(ranks=ranks, samples=samples, 
                                 times=times, throughputs=throughputs))

plt.figure(figsize=(9,8))
plt.plot(scaling_cgpu.ranks, scaling_cgpu.throughputs, 'o-', label='Real data')
plt.xlabel('Number of Summit-GPUs')
plt.ylabel('Training throughput [samples/s]')

plt.plot(scaling_cgpu.ranks, scaling_cgpu.ranks*scaling_cgpu.throughputs[0]/6, '--', label='Ideal')
plt.legend(loc=0);
plt.show()

###########################
## 100 nodes -- Weak Scale
ranks = np.array([6, 12, 24, 48, 96, 192, 384, 600])
results_pattern = os.path.expandvars('../summit_output/weak_scaling/log_%i')

configs, results = [], []
for r in ranks:
    result_dir = results_pattern % r
    configs.append(load_config(result_dir))
    results.append(load_result(result_dir).assign(ranks=r))

samples = np.array([get_num_samples(c) for (c,r) in zip(configs, ranks)])
times = np.array([compute_mean_time(r) for r in results])
throughputs = samples / times

scaling_cgpu = pd.DataFrame(dict(ranks=ranks, samples=samples, 
                                 times=times, throughputs=throughputs))

plt.figure(figsize=(9,8))
plt.plot(scaling_cgpu.ranks, scaling_cgpu.throughputs, 'o-', label='Real data')
plt.xlabel('Number of Summit-GPUs')
plt.ylabel('Training throughput [samples/s]')

plt.plot(scaling_cgpu.ranks, scaling_cgpu.ranks*scaling_cgpu.throughputs[0]/6, '--', label='Ideal')
plt.legend(loc=0);
plt.show()


###########################
## 100 nodes -- Weak Scale
ranks = np.array([6, 12, 24, 48, 96, 192, 384, 600])
results_pattern = os.path.expandvars('../summit_output/weak_scaling/log_dummy_%i')

configs, results = [], []
for r in ranks:
    result_dir = results_pattern % r
    configs.append(load_config(result_dir))
    results.append(load_result(result_dir).assign(ranks=r))

samples = np.array([get_num_samples(c)*r for (c,r) in zip(configs, ranks)])
times = np.array([compute_mean_time(r) for r in results])
throughputs = samples / times

scaling_dummy = pd.DataFrame(dict(ranks=ranks, samples=samples, 
                                  times=times, throughputs=throughputs))


ranks = np.array([6, 12, 24, 48, 96, 192, 384, 600])
results_pattern = os.path.expandvars('../summit_output/weak_scaling/log_%i')

configs, results = [], []
for r in ranks:
    result_dir = results_pattern % r
    configs.append(load_config(result_dir))
    results.append(load_result(result_dir).assign(ranks=r))

samples = np.array([get_num_samples(c) for (c,r) in zip(configs, ranks)])
times = np.array([compute_mean_time(r) for r in results])
throughputs = samples / times

scaling_scratch = pd.DataFrame(dict(ranks=ranks, samples=samples, 
                                    times=times, throughputs=throughputs))


plt.figure(figsize=(9,8))
plt.plot(scaling_scratch['ranks'], scaling_scratch['throughputs'], 'o-', label='Real data')
plt.plot(scaling_dummy['ranks'], scaling_dummy['throughputs'], '^-', label='Dummy data')
plt.xlabel('Number of Summit-GPUs')
plt.ylabel('Training throughput [samples/s]')

plt.plot(scaling_scratch.ranks, scaling_scratch.ranks*scaling_dummy['throughputs'][0]/6, '--', label='Ideal')
plt.legend(loc=0);
plt.show()
