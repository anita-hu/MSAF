## RAVDESS Dataset
This code along with our trained weights reproduces our results on RAVDESS dataset in Table 1 of our paper.

## Setup
Install dependencies
```
pip install opencv-python moviepy librosa sklearn
```
Download the RAVDESS dataset using the bash script
```
bash scripts/download_ravdess.sh <path/to/RAVDESS>
```
Or download the files manually 
- [All Video_Speech_Actors_##.zip files](https://zenodo.org/record/1188976)
- [FacialTracking_Actors_01-24.zip](https://zenodo.org/record/3255102) 

and follow the folder structure below and have .csv files in `landmarks/` (do not modify file names)
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
Download checkpoints folder from [Google Drive](https://drive.google.com/drive/folders/14NqAECoZ58tlpkKtr8FiRtT7j_zOZCYN). 
The following script downloads all pretrained models (unimodal and MSAF) for all 6 folds.
```
bash scripts/download_checkpoints.sh
```

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
