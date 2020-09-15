## RAVDESS Dataset
This code along with our trained weights reproduces our results on RAVDESS dataset in Table 1 of our paper.

## Setup
Install dependencies
```
pip install opencv-python moviepy librosa sklearn
```
Download the RAVDESS dataset and unzip
- [Audio_Speech_Actors_01-24.zip](https://zenodo.org/record/1188976)
- [FacialTracking_Actors_01-24.zip](https://zenodo.org/record/3255102) 

Follow the folder structure below and have .csv files in `landmarks/` (do not modify file names)
```
RAVDESS/
    landmarks/
        .csv landmark files
    Actor_01/
    ...
    Actor_24/
```
Preprocess the dataset using the following
```
python dataset_prep.py --datadir <path/to/RAVDESS>
```
Generated folder structure (do not modify file names)
```
RAVDESS/
    landmarks/
        .csv landmark files
    Actor_01/
    ...
    Actor_24/
    preprocessed/
        Actor_01/
        ...
        Actor_24/
            01-01-01-01-01-01-24.mp4/
                frames/
                    .jpg frames
                audios/
                    .wav raw audio
                    .npy MFCC features
            ...
```
Download checkpoints folder from [Google Drive](https://drive.google.com/drive/folders/1WchLLueLT27Zqeaj4JmhpB6UFGds5wpF?usp=sharing)

## Evaluate
```
python main_msaf.py --datadir <path/to/RAVDESS/preprocessed> \
--checkpointdir checkpoints
```

## Train
```
python main_msaf.py --datadir <path/to/RAVDESS/preprocessed> \ 
--checkpointdir checkpoints --train \
```
All parameters
```
usage: main_msaf.py [-h] [--datadir DATADIR] [--k_fold K_FOLD] [--lr LR]
                    [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                    [--epochs EPOCHS] [--checkpointdir CHECKPOINTDIR] [--no_verbose]
                    [--log_interval LOG_INTERVAL] [--no_save] [--train]
```
