# Phantomware

Phantomware is a geometry-guided data augmentation framework for malware detection under extreme data scarcity scenarios
such as small malware families or emerging malware campaigns. 
It generates diverse synthetic malware representations directly in the latent space of a pretrained malware detector.
Generated samples are then used to augment the training data, improving robustness against unseen malware 
variants without requiring functional malware generation or adversarial optimization.

## Usage

1. Update `target_family.txt` with the name of the target malware family.
2. Populate the `data/real_data/goodware` and `data/real_data/malware` directories with your training data, ensuring malware samples use the .exe file extension while goodware samples remain without any file extension.
3. Populate `training_goodware.csv` and `training_malware.csv` with the filenames of your selected training samples.
4. Run `MalConvGCT_nocatTrain.py` to train the MalConv model on your dataset.
5. Run `embedding_generation.py` to generate synthetic malware representations in the latent space of the pretrained MalConv model.
6. Run `finetuning.py` to fine-tune the pretrained MalConv model on the augmented dataset.

## Contents

### malconv 
This directory contains a Phantomware-compatible version of MalConv2, a malware detector proposed in the AAAI 2021 paper
[Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection](https://arxiv.org/abs/2012.09390).

### phantomware

This directory contains the Phantomware framework. 

## Requirements

 - `torch==2.1.2`
 - `numpy==1.26.0`
 - `pandas==2.1.1`

## Cite

TBA