import pickle 
import torch
from torch import nn, Tensor
import numpy as np

def one_hot_embedding(labels: Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.
    Args:
        labels: class labels, sized [N,].
        num_classes: number of classes.
    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    device = labels.device
    y = torch.eye(num_classes).to(device)
    return y[labels]

def convert_sequences(loaded_data, process_dim):
    event_times_list = loaded_data['timestamps']
    event_types_list = loaded_data['types']
    
    event_times_multivariate = []
    event_types_multivariate = []
    event_intervals_multivariate = []
    event_positions_multivariate = []
    for event_types, event_times in zip(event_types_list, event_times_list):
        event_times_zeros = np.insert(event_times,0,0)
        event_positions = np.arange(0, len(event_times), 1)
        one_seq_times = []
        one_seq_types = []
        one_seq_intervals = []
        one_seq_positions = []

        for event_type in range(process_dim):
            uni_event_time = np.insert(event_times[event_types==event_type],0,0.0)
            uni_event_type = np.insert(event_types[event_types==event_type],0,event_type)
            uni_event_pos = np.insert(event_positions[event_types==event_type],0,-1)

            one_seq_times.append(uni_event_time)
            one_seq_types.append(uni_event_type)
            one_seq_positions.append(uni_event_pos)

        for event_type in range(process_dim):
            one_seq_time = one_seq_times[event_type]
            one_seq_dt = one_seq_time[1:] - one_seq_time[:-1]
            one_seq_dt = np.insert(one_seq_dt,0,0)
            one_seq_intervals.append(one_seq_dt)

        assert np.concatenate(one_seq_positions).max() == (np.concatenate(one_seq_types)!=process_dim).sum() - process_dim - 1
        assert len(one_seq_times) == process_dim == len(one_seq_types) == len(one_seq_intervals) == len(one_seq_positions)
        event_times_multivariate.append(one_seq_times)
        event_types_multivariate.append(one_seq_types)
        event_intervals_multivariate.append(one_seq_intervals)
        event_positions_multivariate.append(one_seq_positions)

    return event_times_multivariate, event_types_multivariate, event_intervals_multivariate, event_positions_multivariate

def process_seq(dataset_dir, sub_datasets, num_samples, process_dim):
    with open(dataset_dir + 'whole_manifold_format.pkl', 'rb') as f:
        data = pickle.load(f)
    print('process dataset...')
    seq_times, seq_types, seq_lengths, seq_intervals = data['timestamps'], data['types'], data['lengths'], data['timeintervals']
    t_max = np.concatenate(seq_times).max()

    event_times_multivariate, event_types_multivariate, event_intervals_multivariate, event_positions_multivariate \
        = convert_sequences(data, process_dim)
    num_samples_ = np.insert(num_samples, 0, 0)
    for i, sub_dataset in enumerate(sub_datasets):
        save_path = dataset_dir + '{}_manifold_format.pkl'.format(sub_dataset)

        with open(save_path, "wb") as f:
            save_data_ = {'timestamps': seq_times[np.sum(num_samples_[:i+1]):np.sum(num_samples_[:i+2])],
                        'types': seq_types[np.sum(num_samples_[:i+1]):np.sum(num_samples_[:i+2])],
                        'intervals': seq_intervals[np.sum(num_samples_[:i+1]):np.sum(num_samples_[:i+2])],
                        'lengths': seq_lengths[np.sum(num_samples_[:i+1]):np.sum(num_samples_[:i+2])],
                        'event_times_multivariate': event_times_multivariate[np.sum(num_samples_[:i+1]):np.sum(num_samples_[:i+2])],
                        'event_types_multivariate': event_types_multivariate[np.sum(num_samples_[:i+1]):np.sum(num_samples_[:i+2])],
                        'event_intervals_multivariate': event_intervals_multivariate[np.sum(num_samples_[:i+1]):np.sum(num_samples_[:i+2])],
                        'event_positions_multivariate': event_positions_multivariate[np.sum(num_samples_[:i+1]):np.sum(num_samples_[:i+2])],
                        't_max': t_max
                        }
            pickle.dump(save_data_,f)
