## RAVDESS Dataset
This code along with our trained weights reproduces our results on RAVDESS dataset in Table 1 of our paper.

## Setup
Install dependencies
```
pip install opencv-python face-recognition moviepy librosa
```
Download the RAVDESS dataset and unzip
- [Audio_Speech_Actors_01-24.zip](https://zenodo.org/record/1188976)

Preprocess the dataset using the following
```
python dataset_prep.py --datadir <path/to/RAVDESS>
```
Generated folder structure (do not modify file names)
```
RAVDESS/
    Actor_01/
    ...
    Actor_24/
    preprocessed/
        Actor_01/
        ...
        Actor_24/
            01-01-01-01-01-01-24.mp4/
                frames/
                audio/
            ...
```

## Evaluate
```
python main_msaf.py --datadir <path/to/RAVDESS> \
--checkpointdir checkpoints \
--no_bad_skel --model msaf --rgb_net i3d --vid_len 32 32
```

## Train
```
python main_msaf.py --datadir <path/to/RAVDESS> \ 
--checkpointdir checkpoints --train \
--ske_cp skeleton_32frames_85.24.checkpoint \
--rgb_net i3d \ 
--rgb_cp i3d_32frames_85.63.checkpoint \
--vid_len 32 32
```
All parameters
```
usage: main_msaf.py [-h] [--rgb_net {resnet,i3d}]
                    [--checkpointdir CHECKPOINTDIR] [--datadir DATADIR]
                    [--ske_cp SKE_CP] [--rgb_cp RGB_CP] [--test_cp TEST_CP]
                    [--num_outputs NUM_OUTPUTS] [--batchsize BATCHSIZE]
                    [--epochs EPOCHS] [--use_dataparallel] [--j NUM_WORKERS]
                    [--modality MODALITY] [--no-verbose] [--no-multitask]
                    [--vid_len VID_LEN [VID_LEN ...]] [--drpt DRPT]
                    [--no_bad_skel] [--no_norm]
                    [--fc_final_preds FC_FINAL_PREDS] [--train]
```
