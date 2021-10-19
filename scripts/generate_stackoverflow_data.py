import pickle
import numpy as np
from process_batch_seq import process_seq
import torch

def convert_stackoverflow(max_length):
    timestamps_list = []
    types_list = []
    lengths_list = []
    timeintervals_list = []
    task = 'stackoverflow'

    file_path = 'data/' + task + '/' + 'whole_dataset' +'.pkl'
    file = torch.load(file_path)
    dim_process = file['num_marks']
    print('dim_process: {} for task: {}'.format(dim_process,task))
    seqs = file['sequences']
    one_seq_num = 0
    for seq in seqs:
        t_start = seq['t_start']
        timestamps =  np.array(seq['arrival_times']) - t_start
        timestamps = timestamps[:max_length]
        types = np.array(seq['marks'])[:max_length]
        timeintervals = np.ediff1d(np.concatenate([[0], timestamps]))
        
        lengths = len(timestamps)
        if lengths == 1:
            one_seq_num += 1
            continue
        timestamps_list.append(np.asarray(timestamps))
        types_list.append(np.asarray(types))
        lengths_list.append(np.asarray(lengths))
        timeintervals_list.append(np.asarray(timeintervals))

    print('one_seq_num: {}'.format(one_seq_num))
    dataset_dir = 'data/' + task + '/'
    save_path = 'data/' + task + '/' + 'whole' +'_manifold_format.pkl'
    with open(save_path, "wb") as f:
        save_data_ = {'timestamps': np.asarray(timestamps_list),
                     'types': np.asarray(types_list),
                     'lengths': np.asarray(lengths_list),
                     'timeintervals': np.asarray(timeintervals_list)
                      }
        pickle.dump(save_data_,f)
    return dataset_dir, dim_process

if __name__ == '__main__':
    max_length = 256
    sub_dataset = ['train', 'val', 'test']
    num_samples = [4633, 700, 1300]
    save_path, process_dim = convert_stackoverflow(max_length)
    process_seq(save_path, sub_dataset, num_samples, process_dim=process_dim)