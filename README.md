This is an official source code for implementation on Extensive Deep Temporal Point Process, which is composed of the following three parts: 

**1. [REVIEW on methods on deep temporal point process](##Reviews)**

**2. [PROPOSITION of a framework on Granger causality discovery](##Granger-causality-framework)**

**3. [FAIR empirical study](##Fair-empirical-study)**

## Reviews
We first conclude the recent research topics on deep temporal point process as four parts:
``· Encoding of history sequence``

``· Relational discovery of events``

``· Formulation of conditional intensity function``

``· Learning approaches for optimization``
<p align="center">
  <img src='./figs/fourparts.PNG' width="1200">
</p>
By dismantling representative methods into the four parts, we list their contributions on temporal point process.

**Methods with the same learning approaches:**
| Methods    | History Encoder | Intensity Function   | Relational Discovery       | Learning Approaches | Released codes                                           |
|------------|-----------------|----------------------|----------------------------|---------------------|----------------------------------------------------------|
| [RMTPP](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf)     | RNN             | Gompertz             | /                          | MLE with SGD        | https://github.com/musically-ut/tf_rmtpp                 |
| [ERTPP](https://arxiv.org/pdf/1705.08982.pdf)      | LSTM            | Gaussian             | /                          | MLE with SGD        | https://github.com/xiaoshuai09/Recurrent-Point-Process   |
| [CTLSTM](https://arxiv.org/pdf/1612.09328.pdf)     | CTLSTM          | Exp-decay + softplus | /                          | MLE with SGD        | https://github.com/HMEIatJHU/neurawkes                   |
| [FNNPP](https://arxiv.org/pdf/1905.09690.pdf)      | LSTM            | FNNIntegral          | /                          | MLE with SGD        | https://github.com/omitakahiro/NeuralNetworkPointProcess |
| [LogNormMix](https://arxiv.org/pdf/1909.12127.pdf) | LSTM            | Log-norm Mixture     | /                          | MLE with SGD        | https://github.com/shchur/ifl-tpp                        |
| [SAHP](https://arxiv.org/pdf/1907.07561.pdf)       | Transformer     | Exp-decay + softplus | Attention Matrix           | MLE with SGD        | https://github.com/QiangAIResearcher/sahp_repo           |
| [THP](https://arxiv.org/pdf/2002.09291.pdf)        | Transformer     | Linear + softplus    | Structure learning         | MLE with SGD        | https://github.com/SimiaoZuo/Transformer-Hawkes-Process  |
| [DGNPP](https://dl.acm.org/doi/pdf/10.1145/3442381.3450135)      | Transformer     | Exp-decay + softplus | Bilevel Structure learning | MLE with SGD        | No available codes until now.                            |

**Methods focusing on learning approaches:**
- Reinforcement learning:
    - [U. Upadhyay, A. De, and M. Gomez Rodriguez, "Deep reinforcement   learning   of   marked   temporal  point processes"](https://papers.nips.cc/paper/2018/file/71a58e8cb75904f24cde464161c3e766-Paper.pdf)
    - [S.  Li,  S.  Xiao,  S.  Zhu,  N.  Du,  Y.  Xie,  and  L.  Song, "Learning temporal point processes via reinforcement learning"](https://papers.nips.cc/paper/2018/file/5d50d22735a7469266aab23fd8aeb536-Paper.pdf)

- Adversarial and discrimitive learning:
    - [J.  Yan,  X.  Liu,  L.  Shi,  C.  Li,  and  H.  Zha,  "Improving maximum   likelihood   estimation   of   temporal   pointprocess  via  discriminative  and  adversarial  learning"](https://www.ijcai.org/Proceedings/2018/0409.pdf)

- Noise contrastive learning:
    - [R. Guo, J. Li, and H. Liu, "Initiator: Noise-contrastiveestimation    for    marked    temporal    point    process"](https://www.ijcai.org/proceedings/2018/0303.pdf)

**Expansions:**
- Spatio-temporal point process:
    - [O.  Maya,  I.  Tomoharu,  K.  Takeshi,  T.  Yusuke,  T.  Hi-royuki,  and  U.  Naonori,  “Deep  mixture  point  pro-cesses"](https://arxiv.org/pdf/1906.08952.pdf)
    - [S. Zhu, S. Li, Z. Peng, and Y. Xie, "Imitation learning ofneural spatio-temporal point processes"](https://arxiv.org/pdf/1906.05467.pdf)
    - [R. T. Q. Chen, B. Amos, and M. Nickel, "Neural spatio-temporal point processes"](https://arxiv.org/pdf/2011.04583.pdf)
- Other applications:
    - [J. Enguehard, D. Busbridge, A. Bozson, C. Woodcock,and N. Y. Hammerla, "Neural temporal point processesfor modelling electronic health records"](https://arxiv.org/pdf/2007.13794.pdf)
    - [E. Nsoesie, M. Marathe, and J. Brownstein, "Forecast-ing peaks of seasonal influenza epidemics"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3712489/)
    - [R. Trivedi, M. Farajtabar, P. Biswal, and H. Zha, “Dyrep: Learning    representations    over    dynamic    graphs”](https://openreview.net/pdf?id=HyePrhR5KX)
    - [S.  Li,  L.  Wang,  X.  Chen,  Y.  Fang,  and  Y.  Song,  "Understanding the spread of covid-19 epidemic: A spatio-temporal point process view"](https://arxiv.org/pdf/2106.13097.pdf)
    - [Q.  JA,  M.  I,  and  N.  A,  "Point  process  methods  inepidemiology:  application  to  the  analysis  of  human immunodeficiency  virus/acquired  immunodeficiency syndrome  mortality  in  urban  areas"](https://pubmed.ncbi.nlm.nih.gov/28555483/)

## Granger causality framework
The workflows of the proposed granger causality framework:
<p align="center">
  <img src='./figs/variationalframe.PNG' width="700">
</p>
Experiments shows improvements in fitting and predictive ability in type-wise intensity modeling settings. And the Granger causality graph can be obtained:
<p align="center">
  <img src='./figs/learned_graph.PNG' width="400" >
</p>
<p align="center">
    <em>Learned Granger causality graph on Stack Overflow</em>
</p>
## Fair empirical study
The results is showed in the Section 6.3. Here we give an instruction on implementation.
### Installation

Requiring packages:
```
pytorch=1.8.0=py3.8_cuda11.1_cudnn8.0.5_0
torchvision=0.9.0=py38_cu111
torch-scatter==2.0.8
```

### Dataset
We provide the MOOC and Stack Overflow datasets in
``./data/``

And Retweet dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1SDxuu9lUTC7gPxhuYZWARrir15VvvOAD?usp=sharing). Download it and copy it into 
``./data/retweet/``

To preprocess the data, run the following commands
```bash
python /scripts/generate_mooc_data.py
python /scripts/generate_stackoverflow_data.py
python /scripts/generate_retweet_data.py
```

### Training
You can train the model with the following commands:
```bash
python main.py --config_path ./experiments/mooc/config.yaml
python main.py --config_path ./experiments/stackoverflow/config.yaml
python main.py --config_path ./experiments/retweet/config.yaml
```

The ``.yaml`` files consist following kwargs:
```
log_level: INFO

data:
  batch_size: The batch size for training
  dataset_dir: The processed dataset directory
  val_batch_size: The batch size for validation and test
  event_type_num: Number of the event types in the dataset. {'MOOC': 97, "Stack OverFlow": 22, "Retweet": 3}

model:
  encoder_type: Used history encoder, chosen in [FNet, RNN, LSTM, GRU, Attention]
  intensity_type: Used intensity function, chosen in [LogNormMix, GomptMix, LogCauMix, ExpDecayMix, WeibMix, GaussianMix] and 
        [LogNormMixSingle, GomptMixSingle, LogCauMixSingle, ExpDecayMixSingle, WeibMixSingle, GaussianMixSingle, FNNIntegralSingle],
        where *Single means modeling the overall intensities
  time_embed_type: Time embedding, chosen in [Linear, Trigono]
  embed_dim: Embeded dimension
  lag_step: Predefined lag step, which is only used when intra_encoding is true
  atten_heads: Attention heads, only used in Attention encoder, must be a divisor of embed_dim.
  layer_num: The layers number in the encoder and history encoder
  dropout: Dropout ratio, must be in 0.0-1.0
  gumbel_tau: Initial temperature in Gumbel-max
  l1_lambda: Weight to control the sparsity of Granger causality graph
  use_prior_graph: Only be true when the ganger graph is given, chosen in [true, false]
  intra_encoding: Whether to use intra-type encoding,  chosen in [true, false]

train:
  epochs: Training epoches
  lr: Initial learning rate
  log_dir: Diretory for logger
  lr_decay_ratio: The decay ratio of learning rate
  max_grad_norm: Max gradient norm
  min_learning_rate: Min learning rate
  optimizer: The optimizer to use, chosen in [adam]
  patience: Epoch for early stopping 
  steps: Epoch numbers for learning rate decay. 
  test_every_n_epochs: 10
  experiment_name: 'stackoverflow'
  delayed_grad_epoch: 10
  relation_inference: Whether to use graph discovery, chosen in [true, false],
        if false, but intra_encoding is true, the graph will be complete.
  
gpu: The GPU number to use for training

seed: Random Seed
```

