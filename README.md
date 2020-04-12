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
- Python 3+
- torchtext 0.4.0+
- Download and extract the dataset [RWTH-PHOENIX-Weather 2014 T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and process it with image2tensor.py

## Usage 

### G2T: Training gloss to text model  

```python
python3 train.py -b 32 -dm 128 -df 512 -n 2 -at 8
```
### E2E: Training end to end model 

```python
python3 train.py -b 32 -dm 128 -df 512 -n 2 -at 8 -e2e
```

### Evaluation

To evaluate the models we use bleu.py, that scores BLEU with n grams from 1 to 4

```python
python3 bleu.py -m model_path -args_with_model_size
```

#### Coded by
- Eloi Marimon Rollant
- Quim Suazo Beneit
- Magda Sztandarska
- Daniel Garcia Tello


