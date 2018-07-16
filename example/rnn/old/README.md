RNN Example
===========
This folder contains RNN examples using low level symbol interface.

## Data
Run `get_sherlockholmes_data.sh` to download Sherlock Holmes data.

## Python

- [lstm.py](lstm.py) Functions for building a LSTM Network
- [gru.py](gru.py) Functions for building a GRU Network
- [lstm_bucketing.py](lstm_bucketing.py) Sherlock Holmes language model by using LSTM
- [gru_bucketing.py](gru_bucketing.py) Sherlock Holmes language model by using GRU
- [char-rnn.ipynb](char-rnn.ipynb) Notebook to demo how to train a character LSTM by using ```lstm.py```


Performance Note:
More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.readthedocs.org/en/latest/faq/env_var.html).
