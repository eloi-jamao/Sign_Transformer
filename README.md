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
- Pytorch 
- Download and extract the dataset [RWTH-PHOENIX-Weather 2014 T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) 

## Usage 

### G2T: Training Gloss to text model  

```python
python3 train.py -e 500 -b 64 -dm 128 -df 512 -n 2 -at 8
```
### S2T: Training Sign to Text models 

#### Using 2+1D Resnet as CNN for feature extraction
```python
python3 train.py -e 'epochs' -b 32 -dm 128 -df 512 -n 2 -at 8 -e2e -w 2 --features-path 'path to extracted features from images' 
```
For faster training we exrtacted the features previously using img2tensor.py and saving. 
See python3 img2tensor.py -h for usage

#### Using 3D Resnet as CNN for feature extraction
```python
python3 Magda you shloud fill this line
```

### Evaluation

To evaluate the models we use bleu.py, that scores BLEU with n grams from 1 to 4

```python
python3 bleu.py -m model_path -args_with_model_size

or 

Magda this line is yours use, copy hgow you evaluated
```

#### Coded by
- Eloi Marimon Rollant
- Quim Suazo Beneit
- Magda Sztandarska
- Daniel Garcia Tello


