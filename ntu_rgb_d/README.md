## NTU RGB+D Dataset
This code along with our trained weights reproduces our results on NTU RGB+D dataset in Table 3 of our paper.

## Setup
Download the NTU RGB+D dataset. Using the following folder structure
```
NTU/
    nturgbd_rgb/
        avi_256x256_30/
            .avi files
    nturgbd_skeletons/
        .skeleton files
```
- Copy all skeleton files to path/NTU/nturgbd_skeletons/ directory
- Change all video clips resolution to 256x256 30fps and copy them to path/NTU/nturgbd_rgb/avi_256x256_30/ directory

Sample bash command to change video clip resolution
```
mkdir avi_256x256_30
for i in *.avi; do ffmpeg -i "$i" -s 256x256 -c:a copy "avi_256x256_30/$i"; done
```
Install dependencies
```
pip install matplotlib opencv-python
```

## Evaluate
To evaluate one checkpoint file
```
python main_msaf.py --datadir <path/to/NTU> \
--checkpointdir checkpoints \
--test_cp msaf_ntu_epoch12_92.24.checkpoint \
--no_bad_skel
```
To evaluate all msaf checkpoints in a folder
```
python main_msaf.py --datadir <path/to/NTU> \
--checkpointdir checkpoints \
--no_bad_skel
```
To evaluate with resnet as the RGB model, download the pretrained checkpoint from [Google Drive](https://drive.google.com/drive/u/0/folders/14tjkHojPH4S7pZnIk4DXXI49I-6mTKqe)
```
python main_msaf.py --datadir <path/to/NTU> \
--checkpointdir checkpoints \
--rgb_net resnet \
--test_cp msaf_ntu_resnet_hcn_epoch10_90.63.checkpoint \
--no_bad_skel \
--vid_len 8 32
```
## Train
```
python main_msaf.py --datadir <path/to/NTU> \
--checkpointdir checkpoints --train
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
If you want to train with resnet as the RGB model, download pretrained checkpoint `rgb_8frames_83.91.checkpoint` from [Google Drive link](https://drive.google.com/drive/folders/1wcIepkmCf2NRfnhXVdoNu6wSxkpZmMNm) and use 
```
python main_msaf.py --datadir <path/to/NTU> \
--checkpointdir checkpoints --train \
--rgb_net resnet \
--rgb_cp rgb_8frames_83.91.checkpoint \
--vid_len 8 32
```

## Reference
This code is built upon the [MMTM github repository](https://github.com/haamoon/mmtm) and the [MFAS github repository](https://github.com/juanmanpr/mfas)

To cite their papers
```
@inproceedings{vaezi20mmtm,
 author = {Vaezi Joze, Hamid Reza and Shaban, Amirreza and Iuzzolino, Michael L. and Koishida, Kazuhito},
 booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
 title = {MMTM: Multimodal Transfer Module for CNN Fusion},
 year = {2020}
}
```
```
@inproceedings{perez2019mfas,
  title={Mfas: Multimodal fusion architecture search},
  author={P{\'e}rez-R{\'u}a, Juan-Manuel and Vielzeuf, Valentin and Pateux, St{\'e}phane and Baccouche, Moez and Jurie, Fr{\'e}d{\'e}ric},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6966--6975},
  year={2019}
}
```
