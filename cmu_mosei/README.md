## CMU-MOSEI Dataset
This code reproduces our results on CMU-MOSEI dataset in Table 2 of our paper.

## Setup
Install dependencies for CMU-MultimodalSDK
```
pip install h5py validators tqdm numpy argparse requests colorama
```
Install library for pretrained BERT model
```
pip install pytorch_pretrained_bert
```
Our dependencies
```
pip install sklearn
```
Download and preprocess the dataset with the following
```
python dataset_prep.py --datadir dataset
```
Generated folder structure (do not modify file names)
```
dataset/  # based on --datadir argument
    csd/  # can delete this folder to save space
        .csd files
    train/
        .npy files
    val/
        .npy files
    test/
        .npy files
```

## Evaluate
```
python main_msaf.py --datadir dataset \
--checkpoint checkpoints/msaf_mosei_epoch6.pth
```

## Train
Basic training command
```
python main_msaf.py --datadir dataset --train
```
All parameters
```
usage: main_msaf.py [-h] [--datadir DATADIR] [--lr LR]
                    [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                    [--epochs EPOCHS] [--checkpoint CHECKPOINT]
                    [--save_path SAVE_PATH] [--no_verbose]
                    [--log_interval LOG_INTERVAL] [--no_save] [--train]
```