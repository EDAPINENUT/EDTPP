from models import EDTPP
import torch
import nni
import os
from pathlib import Path
import numpy as np
import torch
from dataset.dataloader import load_dataset
from models.lib.utils import one_hot_embedding

from models.lib.logger import get_logger
from tqdm import tqdm
from models.lib import optimizers, utils
torch.set_num_threads(4)

class Supervisor:
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self._experiment_name = self._train_kwargs['experiment_name']
        self._log_dir = self._get_log_dir(self, kwargs)
        self._device = torch.device("cuda:{}".format(kwargs.get('gpu')) if torch.cuda.is_available() else "cpu") 
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        self._data, self._seq_lengths, self._max_t, self._granger_graph = load_dataset(device=self._device, **self._data_kwargs)
        self._event_type_num = self._data_kwargs['event_type_num']
        
        if self._train_kwargs['relation_inference'] == False:
            self._granger_graph = torch.ones((self._event_type_num, self._event_type_num))
        
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 5.0)
        self._model = EDTPP(
            event_type_num=self._event_type_num,
            device=self._device,
            **self._model_kwargs
        )

        if self._train_kwargs['relation_inference'] == False and self._model_kwargs['intra_encoding'] == True:
            assert self._granger_graph is not None, ('The prior graph is not provided!')
            self._model = EDTPP(
                event_type_num=self._data_kwargs['event_type_num'],
                prior_graph=self._granger_graph,
                device=self._device,
                **self._model_kwargs
            )
        self._model.to(self._device)
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model(self._epoch_num)

    @staticmethod
    def _get_log_dir(self, kwargs):
        log_dir = Path(kwargs['train'].get('log_dir'))/self._experiment_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        model_path = Path(self._log_dir)/'saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        config = dict(self._kwargs)
        config['model_state_dict'] = self._model.state_dict()
        config['epoch'] = epoch
        model_name = model_path/('epo%d.tar' % epoch)
        torch.save(config, model_name)
        self._logger.info("Saved model at {}".format(epoch))
        return model_name
    
    def load_model(self, epoch_num):
        model_path = Path(self._log_dir)/'saved_model'
        model_name = model_path/('epo%d.tar' % epoch_num)

        assert os.path.exists(model_name), 'Weights at epoch %d not found' % epoch_num

        checkpoint = torch.load(model_name, map_location='cpu')
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch_num))

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)
    
    def _train(self, lr,
               steps, patience=100, epochs=1500, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, lambda_l2=3.0e-4, test_n=5, **kwargs):

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=lambda_l2)
        
        if self._model_kwargs['encoder_type'] == 'Attention':
            self.model_opt = optimizers.NoamOpt(self._model.embed_dim, factor=1, warmup=100, initial_lr=lr, optimizer=optimizer)
        else:
            self.model_opt = optimizers.Opt(optimizer)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_opt.optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)
        
        min_val_loss = float('inf')
        wait = 0

        message = "the number of trainable parameters: " + str(utils.count_parameters(self._model))
        self._logger.info(message)
        self._logger.info('Start training the model ...')
        self._evaluate(dataset='test', verbose=True)     
        for epoch_num in range(self._epoch_num, epochs):

            self._model = self._model.train()

            epoch_train_loss = 0

            train_iterator = self._data['train_loader']
            progress_bar = tqdm(train_iterator, unit="batch")
            
            for _, batch in enumerate(progress_bar):
                self.model_opt.optimizer.zero_grad()

                self._model.learn(batch, epoch=epoch_num)

                loss = self._model.compute_loss(batch, train=True)

                loss.backward()
                self.model_opt.step()

                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
                
                progress_bar.set_postfix(training_loss=loss.item())
                self._logger.debug(loss.item())
                epoch_train_loss += loss.detach()

            train_event_num = torch.sum(self._seq_lengths['train']).float()

            lr_scheduler.step()
            val_event_num, epoch_val_log_loss, epoch_val_ce_loss, epoch_val_ape, epoch_val_top1_acc, epoch_val_top3_acc = self._evaluate(dataset='val')
            nni.report_intermediate_result((epoch_val_log_loss / val_event_num).item())

            if (epoch_num % log_every) == log_every - 1:
                message = '---Epoch.{} Train Negative Overall Log-Likelihood per event: {:5f}. ' \
                    .format(epoch_num, epoch_train_loss / train_event_num)
                self._logger.info(message)

                message = '---Epoch.{} Val Negative Log-Likelihood per event: {:5f}; Cross-Entropy per event: {:5f}; MAPE: {:5f}; Acc_Top1: {:5f}, Acc_Top3: {:5f}   ' \
                    .format(
                        epoch_num,
                        epoch_val_log_loss / val_event_num, 
                        epoch_val_ce_loss / val_event_num, 
                        epoch_val_ape / val_event_num,
                        epoch_val_top1_acc / val_event_num,
                        epoch_val_top3_acc / val_event_num
                        )
                self._logger.info(message)
            

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_event_num, epoch_test_log_loss, epoch_test_ce_loss, epoch_test_ape, epoch_test_top1_acc, epoch_test_top3_acc = self._evaluate(dataset='test', verbose=True)
                message = '---Epoch.{} Val Negative Log-Likelihood per event: {:5f}; Cross-Entropy per event: {:5f}; MAPE: {:5f}; Acc_Top1: {:5f}, Acc_Top3: {:5f}   ' \
                    .format(
                        epoch_num,
                        epoch_test_log_loss / test_event_num, 
                        epoch_test_ce_loss / test_event_num, 
                        epoch_test_ape / test_event_num,
                        epoch_test_top1_acc / test_event_num,
                        epoch_test_top3_acc / test_event_num
                        )
                self._logger.info(message)

            # find the best performance on validation set for epoch selection
            if epoch_val_log_loss / val_event_num < min_val_loss:
                wait = 0
                if save_model:
                    best_epoch = epoch_num
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Negative Log-Likelihood decrease from {:.5f} to {:.5f}, '
                        'saving to {}'.format(min_val_loss, epoch_val_log_loss / val_event_num, model_file_name))
                min_val_loss = epoch_val_log_loss / val_event_num

            # early stopping
            elif epoch_val_log_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num, 'the best epoch is: %d' % best_epoch)
                    break

    def _evaluate(self, dataset, epoch_num=0, load_model=False, verbose=False):
        if load_model == True:
            self.load_model(epoch_num)
        
        epoch_log_loss, epoch_ce_loss, epoch_ape, epoch_top1_acc, epoch_top3_acc = 0, 0, 0, 0, 0
        self._model.eval()
        val_iterator = self._data['{}_loader'.format(dataset)]
        
        hidden_relations = []

        for _, batch in enumerate(val_iterator):
            # self.model_opt.optimizer.zero_grad()
            torch.cuda.empty_cache()
            self._model.evaluate(batch)
            
            ape = self._ape_pred_time(self._model.predict_event_time(max_t=self._max_t), batch.out_dts, batch.out_onehots)

            nll, ce = self._model.compute_loss(batch, train=False)
            top1_acc, top3_acc = \
            self._top_k_acc(self._model.predict_event_type(), batch.out_types, batch.out_onehots, top=1),\
            self._top_k_acc(self._model.predict_event_type(), batch.out_types, batch.out_onehots, top=3)

            epoch_log_loss += nll.detach()
            epoch_ce_loss += ce.detach()
            epoch_ape += ape.detach()
            epoch_top1_acc += top1_acc.detach()
            epoch_top3_acc += top3_acc.detach()
            
            del nll, ce, ape, top1_acc, top3_acc
            self.model_opt.optimizer.zero_grad()
            torch.cuda.empty_cache()

            if 'synthetic' in self._data_kwargs['dataset_dir'] \
                and dataset == 'test' \
                and self._train_kwargs['relation_inference'] == True \
                and self._model_kwargs['intra_encoding'] == True:
                granger_graph_batch = self._model.hidden_adjacency
                hidden_relations.append(granger_graph_batch)
                
        
        if 'synthetic' in self._data_kwargs['dataset_dir'] \
                and dataset == 'test' \
                and self._train_kwargs['relation_inference'] == True \
                and self._model_kwargs['intra_encoding'] == True:
            hidden_relation = torch.stack(hidden_relations, dim=0).mean(dim=0).detach().cpu().numpy()
            true_relation = self._granger_graph
            true_relation[true_relation!=0] = 1
            acc, metrics = utils.calculate_metrics(true_relation.flatten(), hidden_relation.flatten())
            message = 'Test hidden relation acc: {}, \n Other metrics: {}'.format(acc, metrics)
            if verbose == True:
                self._logger.info(message)

        event_num = torch.sum(self._seq_lengths['{}'.format(dataset)]).float()

        return event_num, epoch_log_loss, epoch_ce_loss, epoch_ape, epoch_top1_acc, epoch_top3_acc

    def _test_final_n_epoch(self, n=5):
        model_path = Path(self._log_dir)/'saved_model'
        model_list = os.listdir(model_path)
        import re

        epoch_list = []
        for filename in model_list:
            epoch_list.append(int(re.search(r'\d+', filename).group()))

        epoch_list = np.sort(epoch_list)[-n:]
        loss_list = []
        for i in range(n):
            epoch_num = epoch_list[i]
            self.load_model(epoch_num)
            test_event_num, test_epoch_log_loss, test_epoch_ce_loss, test_epoch_ape, test_epoch_top1_acc, test_epoch_top3_acc = self._evaluate(dataset='test', verbose=True)
            test_loss = test_epoch_log_loss / test_event_num 
            message = '---Epoch.{} Test Negative Log-Likelihood per event: {:5f}; Cross-Entropy per event: {:5f}; MAPE: {:5f}; Acc_Top1: {:5f}, Acc_Top3: {:5f}   ' \
                    .format(
                        epoch_num,
                        test_epoch_log_loss / test_event_num, 
                        test_epoch_ce_loss / test_event_num, 
                        test_epoch_ape / test_event_num,
                        test_epoch_top1_acc / test_event_num,
                        test_epoch_top3_acc / test_event_num
                        )
            self._logger.info(message)
            loss_list.append(test_loss.item())
        test_loss_mean, test_loss_std = np.mean(loss_list), np.var(loss_list)
        message = 'Mean Negative_likelihood on test: {}, Variance: {}'.format(test_loss_mean, test_loss_std)
        nni.report_final_result(test_loss.item())

    def _ape_pred_time(self, pred_time, batch_seq_dt, batch_one_hot):
        # relative absolute error
        # batch_seq_dt = batch_seq_dt.clamp(min=1e-7)
        try:
            if len(pred_time.shape) == 3:
                pred_time = pred_time[:,:,:-1]
                per_event = torch.divide((pred_time.clamp(max=self._max_t) - batch_seq_dt[:,:,None]), batch_seq_dt[:,:,None] + 1e-7)
                mask_event = (per_event.abs() * batch_one_hot).clamp(max=100.0)

            elif len(pred_time.shape) == 2:
                per_event = torch.divide((pred_time.clamp(max=self._max_t) - batch_seq_dt), batch_seq_dt + 1e-7)
                mask_event = (per_event.abs()  * batch_one_hot.sum(dim=-1).bool()).clamp(max=100.0)
            return mask_event.sum()
        except:
            return torch.tensor(-1)

    def _top_k_acc(self, pred_event_prob, batch_seq_type, batch_one_hot, top=5):
        # pred_event_prob: (batch_size, seq_len, event_num)
        # batch_seq_type: (batch_size, seq_len)
        try:
            top_pred = torch.argsort(pred_event_prob, dim=-1, descending=True)[...,:top]
            correct = top_pred.eq(batch_seq_type.unsqueeze(-1).expand_as(top_pred))
            correct_k = correct.view(-1).float().sum(0)
            return correct_k
        except:
            return torch.tensor(-1)

if __name__ =='__main__':
    model = EDTPP(event_type_num=2)
    import numpy as np
    seq_dts = np.random.exponential(size=(64,256)).cumsum(axis=-1)
    seq_types = torch.randint(0,3,(64,256))
    seq_dts_uni = np.random.exponential(size=(64,2,128)).cumsum(axis=-1)
    seq_types_uni = torch.randint(0,3,(64,2,128))
    
    seq_dts = torch.tensor(seq_dts).float()
    seq_dts_uni = torch.tensor(seq_dts_uni).float()
    model(seq_dts, seq_types, seq_dts_uni, seq_types_uni)
    model.compute_loss(seq_dts, seq_types)