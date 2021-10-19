import torch
from models.lib.utils import one_hot_embedding
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch.utils.data as data_utils
from pathlib import Path
import os

def load_dataset(dataset_dir, event_type_num, batch_size, val_batch_size=None, scale_normalization=50.0, device=None, **kwargs):
    print('loading datasets...')

    if val_batch_size == None:
        val_batch_size = batch_size
    
    train_set = SequenceDataset(
        dataset_dir, mode='train', batch_size=batch_size, event_type_num=event_type_num, scale_normalization=scale_normalization, device=device
    )

    validation_set = SequenceDataset(
        dataset_dir, mode='val', batch_size=val_batch_size, event_type_num=event_type_num, scale_normalization=scale_normalization, device=device
    )

    test_set = SequenceDataset(
        dataset_dir, mode='test', batch_size=val_batch_size, event_type_num=event_type_num, scale_normalization=scale_normalization, device=device
    )
    
    max_t_normalization = train_set.max_t
    for dataset in [train_set, validation_set, test_set]:
        setattr(dataset, 'max_t_normalization', max_t_normalization)

    mean_in_train, std_in_train = train_set.get_time_statistics()
    train_set.normalize(mean_in_train, std_in_train)
    validation_set.normalize(mean_in_train, std_in_train)
    test_set.normalize(mean_in_train, std_in_train)

    data = {}
    data['train_loader'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    data['val_loader']  = torch.utils.data.DataLoader(validation_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate)
    data['test_loader'] = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate)

    max_t = max([train_set.max_t, validation_set.max_t, test_set.max_t])/max_t_normalization*scale_normalization \
        if scale_normalization != 0 else max([train_set.max_t, validation_set.max_t, test_set.max_t])
    
    if hasattr(train_set, 'granger_graph'):
        granger_graph = train_set.granger_graph
    else: granger_graph = None

    return data, {'train':train_set.seq_lengths, 'val':validation_set.seq_lengths, 'test': test_set.seq_lengths}, max_t, granger_graph


class SequenceDataset(data_utils.Dataset):
    """Dataset class containing variable length sequences.

    Args:
        delta_times: Inter-arrival times between events. List of variable length sequences.

    """
    def __init__(self, dataset_dir, mode, batch_size, event_type_num, device=None, scale_normalization=50.0):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = device

        self.file = dataset_dir + mode + '_manifold_format.pkl'
        self.event_type_num = event_type_num
        self.bs = batch_size
        self.scale_normalization = scale_normalization

        if os.path.exists(dataset_dir + '/granger_graph.npy'):
            self.granger_graph = np.load(dataset_dir + '/granger_graph.npy')

        with open(self.file, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.process_data()


    def load_processed(self):
        with open(self.processed_path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.in_times, self.out_times, self.in_dts, self.out_dts, self.in_types, self.out_types, \
            self.in_multi_times, self.in_multi_types, self.in_multi_dts, self.in_multi_positions = \
                self.dataset['in_times'], self.dataset['out_times'], self.dataset['in_dts'], self.dataset['out_dts'],\
                self.dataset['in_types'], self.dataset['out_types'], self.dataset['in_multi_times'], self.dataset['in_multi_types'],\
                self.dataset['in_multi_dts'], self.dataset['in_multi_positions']


    def process_data(self):
        # print('processing dataset and saving in {}...'.format(self.processed_path))

        self.seq_times, self.seq_types, self.seq_lengths, self.seq_dts, self.event_times_multivariate, \
        self.event_types_multivariate, self.event_intervals_multivariate, self.event_positions_multivariate, self.max_t = \
            self.data['timestamps'], self.data['types'], self.data['lengths'], self.data['intervals'], self.data['event_times_multivariate'],\
            self.data['event_types_multivariate'], self.data['event_intervals_multivariate'], self.data['event_positions_multivariate'], self.data['t_max']
        
        self.max_t = np.concatenate(self.data['timestamps']).max()
        self.seq_lengths = torch.Tensor(self.seq_lengths)

        self.in_times = [torch.Tensor(t[:-1]) for t in self.seq_times]
        self.out_times = [torch.Tensor(t[1:]) for t in self.seq_times]
        
        self.in_dts = [torch.Tensor(dt[:-1]) for dt in self.seq_dts]
        self.out_dts = [torch.Tensor(dt[1:]) for dt in self.seq_dts]

        self.in_types = [torch.Tensor(m[:-1]) for m in self.seq_types]
        self.out_types = [torch.Tensor(m[1:]) for m in self.seq_types]

        self.validate_times()

        self.in_multi_times = [[torch.Tensor(t) for t in ts] for ts in self.event_times_multivariate] 
        self.in_multi_types = [[torch.Tensor(m) for m in ms] for ms in self.event_types_multivariate] 
        # self.in_multi_dts = [[torch.Tensor(t) for t in ts] for ts in self.event_intervals_multivariate] 
        self.in_multi_positions = [[torch.Tensor(p) for p in ps] for ps in self.event_positions_multivariate] 

        # self.dataset = {
        #     'in_times': self.in_times,
        #     'out_times': self.out_times,
        #     'in_dts': self.in_dts,
        #     'out_dts': self.out_dts,
        #     'in_types': self.in_types,
        #     'out_types': self.out_types,
        #     'in_multi_times': self.in_multi_times,
        #     'in_multi_types': self.in_multi_types,
        #     'in_multi_dts': self.in_multi_dts,
        #     'in_multi_positions': self.in_multi_positions,
        #     'max_t': self.max_t
        # }


    @property
    def num_series(self):
        return len(self.in_times)

    def get_time_statistics(self):
        flat_in_times = torch.cat(self.in_times)
        return flat_in_times.mean(), flat_in_times.std()
    
    def validate_times(self):
        if len(self.in_times) != len(self.out_times):
            raise ValueError("in_times and out_times have different lengths.")

        for s1, s2, s3, s4 in zip(self.in_times, self.out_times, self.in_types, self.out_types):
            if len(s1) != len(s2) or len(s3) != len(s4):
                raise ValueError("Some in/out series have different lengths.")
            if s3.max() >= self.event_type_num or s4.max() >= self.event_type_num:
                raise ValueError("Marks should not be larger than number of classes.")

    def normalize(self, mean_in=None, std_in=None):
        """Apply mean-std normalization to times."""
        if mean_in is None or std_in is None:
            mean_in, std_in = self.get_mean_std_in()
        self.in_times = [(t - mean_in) / std_in for t in self.in_times]
        self.in_dts = [(t - mean_in) / std_in for t in self.in_dts]
        # self.in_multi_dts = [[(t - mean_in) / std_in for t in ts] for ts in self.in_multi_dts]

        if self.scale_normalization != 0:
            self.out_times = [t / self.max_t_normalization * self.scale_normalization for t in self.out_times]
            self.out_dts= [t / self.max_t_normalization * self.scale_normalization for t in self.out_dts]

        return self

    def get_mean_std_in(self):
        """Get mean and std of in_times."""
        flat_in_times = torch.cat(self.in_times)
        return flat_in_times.mean(), flat_in_times.std()

    def get_mean_std_out(self):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(self.out_times)
        return flat_out_times.mean(), flat_out_times.std()

    def get_log_mean_std_out(self):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(self.out_times).log()
        return flat_out_times.mean(), flat_out_times.std()

    def __getitem__(self, key):
        return self.in_times[key], self.out_dts[key], self.in_types[key], self.out_types[key], self.seq_lengths[key], \
             self.in_multi_times[key], self.in_multi_types[key], self.in_multi_positions[key], self.event_type_num, self.device

    def __len__(self):
        return self.num_series

    def __repr__(self):
        return f"SequenceDataset({self.num_series})"

def collate(batch):

    def pad_multivariate(multi_seq, padding_value=0):
        multi_seq = [torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value).permute(1,0) for seq in multi_seq]
        return torch.nn.utils.rnn.pad_sequence(multi_seq, batch_first=True, padding_value=padding_value).permute(0,2,1)
    device = batch[0][9]
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    in_times = [item[0] for item in batch]
    out_dts = [item[1] for item in batch]
    in_types = [item[2] for item in batch]
    out_types = [item[3] for item in batch]

    seq_lengths = torch.Tensor([item[4] for item in batch])
    in_multi_times = [item[5] for item in batch]
    in_multi_types = [item[6] for item in batch]
    in_multi_positions = [item[7] for item in batch]
    event_type_num = batch[0][8]

    in_times = torch.nn.utils.rnn.pad_sequence(in_times, batch_first=True, padding_value=0.0)
    out_dts = torch.nn.utils.rnn.pad_sequence(out_dts, batch_first=True, padding_value=0.0)
    in_types = torch.nn.utils.rnn.pad_sequence(in_types, batch_first=True, padding_value=event_type_num)
    out_types = torch.nn.utils.rnn.pad_sequence(out_types, batch_first=True, padding_value=event_type_num)

    in_multi_times = pad_multivariate(in_multi_times, padding_value=0.0)
    in_multi_types = pad_multivariate(in_multi_types, padding_value=event_type_num)
    in_multi_positions = pad_multivariate(in_multi_positions, padding_value=-1)
    
    out_onehots = one_hot_embedding(out_types, event_type_num + 1)
    return Batch(
        in_times.to(device), 
        in_types.to(device), 
        in_multi_times.to(device), 
        in_multi_types.to(device), 
        in_multi_positions.to(device), 
        seq_lengths.to(device), 
        out_dts.to(device), 
        out_types.to(device),
        out_onehots.to(device)
        )


class Batch():
    def __init__(self, in_times, in_types, in_multi_times, in_multi_types, in_multi_positions, seq_lengths, out_dts, out_types, out_onehots):
        self.in_times = in_times
        self.in_types = in_types.long()
        self.in_multi_times = in_multi_times
        self.in_multi_types = in_multi_types.long()
        self.in_multi_positions = in_multi_positions.long()
        self.seq_lengths = seq_lengths
        self.out_dts = out_dts
        self.out_types = out_types.long()
        self.out_onehots = out_onehots.long()