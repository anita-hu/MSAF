## NTU RGB+D Dataset
This code reproduces our results on NTU RGB+D dataset in Table 3 of our paper.

### Setup
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
Dependencies
```
pip install matplotlib opencv-python
```

### Evaluate
```
python main_msaf.py --datadir <path/to/NTU> \
--checkpointdir checkpoints \
--no_bad_skel --model msaf --rgb_net i3d --vid_len 32 32
```

### Train
```
python main_msaf.py --datadir <path/to/NTU> \ 
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

### Reference
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
