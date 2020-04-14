# Transformer Neural Sign Language Translation

Our work is based on the results and dataset from the [Neural Sign Language Translation](https://www-i6.informatik.rwth-aachen.de/publications/download/1064/CamgozCihanHadfieldSimonKollerOscarNeyHermannBowdenRichard--NeuralSignLanguageTranslation--2018.pdf)

The code used was an adapted version of the [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) from Harvard NLP:

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

## Requirements
- Python 3
- pip install -r requirements.txt
- Download and extract the dataset [RWTH-PHOENIX-Weather 2014 T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

## Usage

### G2T: Training Gloss to text model

The best model obtained in this step is a baseline for our further experiments

```python
python3 train.py -e 500 -b 64 -dm 128 -df 512 -n 2 -at 8
```
### S2T: Training Sign to Text models

#### Using 2+1D Resnet as CNN for feature extraction
```python
python3 train.py -e 'epochs' -b 32 -dm 128 -df 512 -n 2 -at 8 -e2e -w 2 -features-path 'path to extracted features from images'
```
For faster training we extracted the features previously using img2tensor.py and saving. See for usage:
```
python3 img2tensor.py -h
```
To train the Sign to text model without doing this process beforehand, it is possible to train only extracting the dataset and using the frames. To do so, comment/uncomment the following lines:
Line 67 from DataLoader.py
```
else:
    #clips = self.make_clips(img_fold, self.long_clips, self.window_clips) #Uncomment this line
    clips = torch.load(img_fold[:-1]) #Comment this other one
    return (clips, label)
```
#### Using 3D Resnet as CNN for feature extraction

```python
python3 S2T/3D/s2t_train.py -e 'epochs' -b 1 -n 2 -at 8 -w 2 -m 'path to pretrained 3d resnet' -f 'path to dataset' -o 'output path'
```
*pretrained model resnet-34-kinetics.pth used in this work can be found [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M)
*the feature extraction network (3D resnet) will be initialized with weights coming from resnet-34-kinetics.pth, pretrained on Kinetics dataset and trained jointly for the final task

### Evaluation

To evaluate the models we use bleu.py, that scores BLEU with n grams from 1 to 4

```python
python3 bleu.py -m model_path -args_with_model_size

or

python3 S2T/3D/s2t_bleu.py -m model_path -n 2 -at 8 -w 2 -f 'path to dataset'
```

#### Coded by
- Eloi Marimon Rollant
- Quim Suazo Beneit
- Magda Sztandarska
- Daniel Garcia Tello
