import pickle
import numpy as np
from process_batch_seq import process_seq
def convert_retweet(max_length):
    timestamps_list = []
    types_list = []
    lengths_list = []
    timeintervals_list = []
    task = 'retweet'

    file_path = 'data/' + task + '/' + 'whole_dataset' +'.pkl'
    with open(file_path, 'rb') as f:
        file = pickle.load(f, encoding='latin1')
        dim_process = file['dim_process']
        print('dim_process: {} for task: {}'.format(dim_process,task))
        seqs = file['data']
        one_seq_num = 0
        for seq in seqs:
            timestamps = []
            types = []
            timeintervals = []
            for event in seq:
                event_type = event['type_event']
                event_timestamp = event['time_since_start']
                event_timeinterval = event['time_since_last_event']

                timestamps.append(event_timestamp[:max_length])
                types.append(event_type[:max_length])
                timeintervals.append(event_timeinterval[:max_length])
            lengths = len(seq)
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
    num_samples = [20000, 2000, 2000]
    save_path, process_dim = convert_retweet(max_length)
    process_seq(save_path, sub_dataset, num_samples, process_dim=process_dim)