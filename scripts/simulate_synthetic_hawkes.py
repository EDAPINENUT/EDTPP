import matplotlib.pyplot as plt
import numpy as np

from tick.base import TimeFunction
from tick.hawkes import SimuHawkes, HawkesKernelExp, HawkesKernelTimeFunc, HawkesKernelSumExp, HawkesKernelPowerLaw
from tick.plot import plot_point_process
import pickle
from tqdm import tqdm
from process_batch_seq import process_seq
import os

def cos_decay_ker(t_axis, lamb, beta):
    t_axis = np.array(t_axis)
    return t_axis, beta * np.abs(np.cos(3*t_axis)) * np.exp(-lamb*t_axis)

def simulate_graph(num_node, probas, ker_num):
    graph_msk = np.random.multinomial(1, probas, num_node*num_node)
    graph_msk = np.array([np.where(r==1)[0][0] for r in graph_msk])
    graph_msk = graph_msk.reshape(num_node, num_node)
    graph_msk = graph_msk + np.identity(num_node)
    graph_msk[graph_msk>(ker_num-1)] = ker_num - 1
    return graph_msk

def simulate_multivariate_hawkes(num_node, num_samples, seed=2021, cut_ratio=1/5):
    np.random.seed(seed)
    ker_num = 5
    proba = [cut_ratio]
    kernel_prob = (1 - cut_ratio)/(ker_num - 1)
    for i in range(ker_num-1):
        proba.append(kernel_prob) 
    graph_idx = simulate_graph(num_node, proba, ker_num).flatten()
    print('simulated graph:\n', graph_idx.reshape(num_node, num_node))
    t_values = np.array([0, 1, 2], dtype=float)
    y_values = np.array([0, 0, 0], dtype=float)
    tf_uncorr = TimeFunction([t_values, y_values], inter_mode=TimeFunction.InterLinear,
                    dt=0.1)
    kernel_0 = HawkesKernelTimeFunc(tf_uncorr)

    t_values = np.linspace(0,10,100)
    t_values, y_values = cos_decay_ker(t_values, 0.25, 0.1)
    tf1 = TimeFunction([t_values, y_values],
                    inter_mode=TimeFunction.InterLinear, dt=0.1)
                    
    kernel_1 = HawkesKernelTimeFunc(tf1)
    kernel_2 = HawkesKernelExp(.08, .4)
    kernel_3 = HawkesKernelSumExp([.01,.03,.05], [.8,.6,.4])
    kernel_4 = HawkesKernelPowerLaw(0.1, 0.5, 2.5)

    ker_list = np.array([kernel_0, kernel_1, kernel_2, kernel_3, kernel_4])
    graph_ker = ker_list[graph_idx.astype(int)].reshape(num_node, num_node)
    graph_ker = graph_ker.tolist()
    num_sample = sum(num_samples)
    timestamps_list = []
    types_list = []
    lengths_list = []
    timeintervals_list = []
    print('start preparing {} set...'.format('whole'))
    progress_bar = tqdm(range(num_sample), unit="sample")
    
    for seed in progress_bar:
        
        hawkes = SimuHawkes(
            kernels=graph_ker,
            baseline=[1.0 for i in range(num_node)], verbose=False, seed=seed)

        run_time = 40
        dt = 0.01
        hawkes.track_intensity(dt)
        hawkes.end_time = run_time
        hawkes.simulate()
        time_stamps = hawkes.timestamps
        i = 0
        seq_types = []
        for time_stamp in time_stamps:
            seq_type = np.array([i for j in range(len(time_stamp))])
            seq_types.append(seq_type)
            i += 1
        time_stamps = np.concatenate(time_stamps)
        seq_types = np.concatenate(seq_types)
        idx = np.argsort(time_stamps)
        time_stamps = time_stamps[idx]
        seq_types = seq_types[idx]
        
        timestamps_list.append(time_stamps)
        types_list.append(seq_types)
        lengths_list.append(len(time_stamps))
        time_stamps = np.insert(time_stamps,0,0)
        timeintervals_list.append(time_stamps[1:] - time_stamps[:-1])
    print('max_length : ', np.max(np.asarray(lengths_list)), 'min_length:', np.min(np.asarray(lengths_list)), 'mean_length:', np.mean(np.asarray(lengths_list)), \
        'num_seqence:', len(np.asarray(lengths_list)))
    save_path = 'data/synthetic_{}_{}/'.format(num_node, cut_ratio) 
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    with open(save_path + 'whole_manifold_format.pkl', "wb") as f:
        save_data_ = {'timestamps': np.asarray(timestamps_list),
                    'types': np.asarray(types_list),
                    'lengths': np.asarray(lengths_list),
                    'timeintervals': np.asarray(timeintervals_list)
                    }
        pickle.dump(save_data_,f)
    np.save(save_path + 'granger_graph', graph_idx.reshape(num_node,num_node))

    fig, ax = plt.subplots(hawkes.n_nodes, 1, figsize=(14, 8))
    plot_point_process(hawkes, t_max=run_time, ax=ax)

    plt.savefig('intensity.png')
    return save_path

if __name__ == '__main__':
    sub_dataset = ['train', 'val', 'test']
    num_samples = [4000, 800, 1200]
    num_nodes = [5]
    cut_ratios= [0.2, 0.4, 0.6]
    for num_node in num_nodes:
        for cut_ratio in cut_ratios:
            dataset_path = 'data/synthetic_{}_{}/'.format(num_node, cut_ratio) #simulate_multivariate_hawkes(num_node, num_samples, cut_ratio=cut_ratio)
            process_seq(dataset_path, sub_dataset, num_samples, process_dim=num_node)