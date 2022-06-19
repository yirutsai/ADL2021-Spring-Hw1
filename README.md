# Homework 1 ADL NTU 110 Spring
## Environment
```
conda create -n <env_name> python=3.8
conda activate <env_name>
pip install -r requirements.txt
```
### Important versions of packages
```
python=3.8
pytorch=1.7.1
seqeval=1.2.2
```
## Preprocessing
```
bash preprocess.sh
```
## Intent detection
### Training
```
python3 train_intent.py
```
* data_dir: Directory to the dataset.
* cache_dir: Directory to the preprocessed caches.
* ckpt_dir: Directory to save the model file.
* hidden_size: hidden state dimension of RNN-based layers.
* num_layers: number of layers.
* dropout: Model dropout rate.
* attn: Do self-attention or not.
* batch_size: Batch size.
* lr: Learning rate.
* netType: Which RNN-based layer we use.(default=LSTM,options:[LSTM,RNN,GRU])


### Predicting
```
python3 test_intent.py --test_file <file path you want to test> --ckpt_path <path of your pre-trained model>
```

### Reproduce(Kaggle)
#### Just predict
```
bash download.sh
bash intent_cls.sh <path of test.json> <path of predict_out.csv>
```
#### Training
```
python3 train_intent.py --attn --use_scheduler --netType GRU --pack --dropout 0.5
```
## Slot tagging
### Training
```
python3 train_slot.py
```
* data_dir: Directory to the dataset.
* cache_dir: Directory to the preprocessed caches.
* ckpt_dir: Directory to save the model file.
* hidden_size: hidden state dimension of RNN-based layers.
* num_layers: number of layers.
* dropout: Model dropout rate.
* attn: Do self-attention or not.
* batch_size: Batch size.
* lr: Learning rate.
* netType: Which RNN-based layer we use.(default=LSTM,options:[LSTM,RNN,GRU])


### Predicting
```
python3 test_slot.py --test_file <file path you want to test> --ckpt_path <path of your pre-trained model>
```

### Reproduce(Kaggle)
#### Just predict
```
bash download.sh
bash slot_tag.sh <path of test.json> <path of predict_out.csv>
```
#### Training
```
python3 train_slot.py --pack --netType LSTM --grad_clip 5 --dropout 0.45
```

