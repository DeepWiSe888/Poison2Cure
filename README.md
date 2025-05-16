# Poison to Cure: Privacy-preserving Wi-Fi Multi-User Sensing via Data Poisoning

Welcome to Poison2Cure! This repository contains the resource code to implement Poison2Cure system (MobiCom'25).


## Contents
[Introduction](#introduction)

[Getting Started](#getting-started)

[Workflow](#workflow)

[Evaluation](#evaluation)


## Introduction

Poison2Cure (P2C) is the first semantic-level privacy-preserving framework for Wi-Fi human sensing systems, with full compatibility to any underlying hardware. The innovation behind Poison2Cure lies in feeding poisoned training data from (privacy-sensitive) users to the neural model for Wi-Fi sensing, degrading only the sensing for private activities while retaining that for regular ones.


## Getting Started

### Environment and Hardware (optional)
Ubuntu 22.04.2 LTS

5.15.0-94-generic kernel

Python 3.8.16

CUDA Version: 11.7

NVIDIA RTX A5000


### Install

1. Download the [Source Code](https://github.com/DeepWiSe888/Poison2Cure#) and [Dataset](https://zenodo.org/records/15010688). Make sure to place the two projects from the source code in the same folder.

2. Install Python: Please ensure Python 3.8.16 is installed on your computer. You can also download the Python source from the [official website](https://www.python.org/).

3. Set Up Virtual Environment: It is recommended to set up a virtual environment to ensure a clean and isolated environment for P2C implementation. Tools like **conda** can be used for this purpose. Make sure to activate your virtual environment before proceeding.

4. Install the necessary packages: We provide the requirements.txt in source code. You can install them by ```pip install -r requirements.txt```.



## Workflow

1. Preparation: Train a human activity recognition model as the attack target.

- Open ```GRU_classifier``` $\rightarrow$ ```mian_GRU.py```, and change the path in line 28 to the folder path of **data_for_train** in the Dataset.

- Run ```GRU_classifier``` $\rightarrow$ ```mian_GRU.py```.

2. Case 1: Fine-tune the model using the poisoned dataset.

- Open ```WitcherBrew``` $\rightarrow$ ```forest``` $\rightarrow$ ```options.py```, and change the path in line 16 to the folder path of **data_for_poison** in the Dataset. Alternatively, you can specify the path using the ```--dataset_dir``` command.
- Run ```WitcherBrew``` $\rightarrow$ ```brew_poison.py```.
- ```WitcherBrew``` $\rightarrow$ ```log_file``` will generate two txt files: ```cleanvalid_log.txt``` and ```poisonedvalid_log.txt```. These represent our results.

3. Case 2: Fine-tune the model using the poisoned dataset under frequency and power bounds.

- Run ```WitcherBrew``` $\rightarrow$ ```brew_poison_constraint.py```.
- ```WitcherBrew``` $\rightarrow$ ```log_file_const``` will generate two txt files: ```cleanvalid_log.txt``` and ```poisonedvalid_log.txt```. These represent our results.


4. Case 3: Fine-tune the model with an unknown architecture using the poisoned dataset.


- Since estimating the model under an unknown architecture requires multiple surrogate models, we have prepared these models in ```WitcherBrew``` $\rightarrow$ ```example```.
- Run ```WitcherBrew``` $\rightarrow$ ```brew_poison_ensemble.py```.
- ```WitcherBrew``` $\rightarrow$ ```log_file_ens``` will generate two txt files: ```cleanvalid_log.txt``` and ```poisonedvalid_log.txt```. These represent our results.



## Evaluation

1. Description

- ```cleanvalid_log.txt```: The testing results after fine-tuning the model with the clean dataset.

- ```poisonedvalid_log.txt```: The testing results after fine-tuning the model with the poisoned dataset.

- ```priv_risk_accuracy```: The recognition rate of privacy activities.

- ```auth_accuracy```: The recognition rate of authorized (non-privacy) activities.

2. Expected results

- In the testing results of the model trained with the clean dataset, the recognition rate for authorized activities is above 80\%, while the recognition rate for privacy activities is around 28\%. However, users expect the recognition rate for privacy activities to approach 0. With the poisoned dataset, the recognition rate for privacy activities is successfully reduced to about 5% (Case 1 and Case 2), and even under an unknown model architecture (Case 3), it remains as low as around 13% with only 5 surrogate models, while the recognition rate for authorized activities stays above 80%. These results demonstrate that our algorithm can successfully preserve the activities that users wish to remain unrecognized through poisoning.
