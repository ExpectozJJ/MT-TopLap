# MT-TopLap

<div align='center'>
 
<!-- [![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.12508&color=B31B1B)](https://www.google.com/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/YOUR_DOI.svg)](https://doi.org/10.5281/zenodo.10892800)

</div>

**Title** - Potential mutations from animal-adapted SARS-CoV-2 that strengthen infectivity on humans

**Authors** - JunJie Wee, Jiahui Chen and Guo-wei Wei

---

## Table of Contents

- [MT-TopLap](#topoformer)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Datasets](#datasets)
  - [Preparing Topologial Sequence](#preparing-topologial-sequence)
  - [Fine-Tuning Procedure for Customized Data](#fine-tuning-procedure-for-customized-data)
  - [Results](#results)
      - [Pretrained models](#pretrained-models)
      - [Finetuned models and performances](#finetuned-models-and-performances)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

---

## Introduction
The COVID-19 pandemic has resulted in a significant SARS-CoV-2 virus evolution, providing a notable advantage in infecting humans using its subvariants. However, such an advantage may not be universal when the virus infect animals. In this work, we hypothesize that as the virus evolution is adapted on animals, it may not necessarily evolve to become more infectious to humans. Nevertheless, there is still a possibility that the virus undergoes a gain-of-function mutation, accelerating virus evolution on humans after evolving on animals. Here, we propose a multitask deep learning model MT-TopLap which simultaneously trains on several deep mutational scanning datasets. MT-TopLap is also validated and finetuned with the largest protein-protein interactions database, SKEMPI 2.0. The resulting MT-TopLap model predicts binding free energy changes upon mutation for RBD binded with human, cat, bat, deer and hamster ACE2 complexes. By comparing the binding free energy changes upon mutation across the different RBD-ACE2s, we identified key RBD mutations such as Q498H in animal-adapted SARS-CoV-2 and R493K in hamster-adapated BA.2 variant having high likelihood to accelerate virus evolution on humans.

> **Keywords**: COVID-19, SARS-CoV-2, BA.2, deep mutational scanning, infectivity, multitask deep learning

---

## Model Architecture

The multitask deep learning architecture of MT-TopLap model is shown below.

![Model Architecture](MT-TopLap.png)

Further explain the details in the [paper](https://github.com/ExpectozJJ/MT-TopLap), providing context and additional information about the architecture and its components.

---

## Getting Started

### Prerequisites

- fair-esm                  2.0.0
- numpy                     1.23.5
- scipy                     1.11.3
- torch                     2.1.1
- pytorch-cuda              11.7
- scikit-learn              1.3.2
- python                    3.10.12

### Installation

```
git clone https://github.com/ExpectozJJ/MT-TopLap
```

---

## Datasets

A brief introduction about the benchmarks.

| | Datasets                    | Training Set                 | Test Set                                             |
|-|-----------------------------|------------------------------|------------------------------                        |
|Pre-training | Combind PDBbind |19513 [RowData]([www](http://www.pdbbind.org.cn/)), [TopoFeature_small](https://weilab.math.msu.edu/Downloads/TopoFormer/TopoFeature_small.npy), [TopoFeature_large](https://weilab.math.msu.edu/Downloads/TopoFormer/TopoFeature_large.npy)  |                                          |
|Finetuning   | CASF-2007       |1105  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 195 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
|             | CASF-2013       |2764  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 195 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
|             | CASF-2016       |3772  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 285 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
|             | PDB v2016       |3767  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 290 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
|             | PDB v2020       |18904 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)<br> (exclude core sets)| 195 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)<br>(CASF-2007 core set) |
|             |                 |                              | 195 <br>(CASF-2013 core set) |
|             |                 |                              | 285 <br>(CASF-2016 core set) |
|             |                 |                              | 285 <br>(v2016 core set)     |

- RowData: the protein-ligand complex structures. From PDBbind
- TopoFeature: the topological embedded features for the protein-ligand complex. All features are saved in a dict, which `key` is the protein ID, and `value` is the topological embedded features for corresponding complex. The downloaded file is .zip file, which contains two file (1) `TopoFeature_large.npy`: topological embedded features with a filtration parameter ranging from 0 to 10 and incremented in steps of 0.1 \AA; (2) `TopoFeature_small.npy`: topological embedded features with a filtration parameter ranging from 2 to 12 and incremented in steps of 0.2 \AA; 
- Label: the .csv file, which contains the protein ID and corresponding binding affinity.

|    Task   | Datasets      | Description |
|-----------|----------     |-------------|
| Screening | LIT-PCBA      | 3D poses for all 15 targets. [Download (13GB)](https://weilab.math.msu.edu/Downloads/TopoFormer/LIT-PCBA_dock_pose.zip)|
|           | PDBbind-v2013 | 3D poses. Download from https://weilab.math.msu.edu/AGL-Score|
| Docking   | CASF-2007,2013| 3D poses. Download from https://weilab.math.msu.edu/AGL-Score |
---
## Preparing Topologial Sequence

```shell
# get the usage
python ./code_pkg/main_potein_ligand_topo_embedding.py -h

# examples
python ./code_pkg/main_potein_ligand_topo_embedding.py --output_feature_folder "../examples/output_topo_seq_feature_result" --protein_file "../examples/protein_ligand_complex/1a1e/1a1e_pocket.pdb" --ligand_file "../examples/protein_ligand_complex/1a1e/1a1e_ligand.mol2" --dis_start 0 --dis_cutoff 5 --consider_field 20
```


## Fine-Tuning Procedure for Customized Data

```shell
bs=32 # batch size
lr=0.00008  # learning rate
ms=10000  # max training steps
fintuning_python_script=./code_pkg/topt_regression_finetuning.py
model_output_dir=./outmodel_finetune_for_regression
mkdir $model_output_dir
pretrained_model_dir=./pretrained_model
scaler_path=./code_pkg/pretrain_data_standard_minmax_6channel_large.sav
validation_data_path=./CASF_2016_valid_feat.npy
train_data_path=./CASF_2016_train_feat.npy
validation_label_path=./CASF2016_core_test_label.csv
train_label_path=./CASF2016_refine_train_label.csv

# finetune for regression on one GPU
CUDA_VISIBLE_DEVICES=1 python $fintuning_python_script --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 --num_train_epochs 100 --max_steps $ms --per_device_train_batch_size $bs --base_learning_rate $lr --output_dir $model_output_dir --model_name_or_path $pretrained_model_dir --scaler_path $scaler_path --validation_data $validation_data_path --train_data $train_data_path --validation_label $validation_label_path --train_label $train_label_path --pooler_type cls_token --random_seed 1234 --seed 1234;
```


```shell
# script for no validation data and validation label
# docking and screening
bs=32 # batch size
lr=0.0001  # learning rate
ms=5000  # max training steps
fintuning_python_script=./code_pkg/topt_regression_finetuning_docking.py
model_output_dir=./outmodel_finetune_for_docking
mkdir $model_output_dir
pretrained_model_dir=./pretrained_model
scaler_path=./code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav
train_data_path=./train_feat.npy
train_label_path=./train_label.csv

# finetune for regression on one GPU
CUDA_VISIBLE_DEVICES=1 python $fintuning_python_script --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 --num_train_epochs 100 --max_steps $ms --per_device_train_batch_size $bs --base_learning_rate $lr --output_dir $model_output_dir --model_name_or_path $pretrained_model_dir --scaler_path /$scaler_path --train_data $train_data_path --train_label $train_label_path --validation_data None --validation_label None --train_val_split 0.1 --pooler_type cls_token --random_seed 1234 --seed 1234 --specify_loss_fct 'huber';
```


---

## Results

#### Pretrained models
- Pretrained TopoFormer model large. [Download](https://weilab.math.msu.edu/Downloads/TopoFormer/TopoFormer_s_pretrained_model.zip)
- Pretrained TopoFormer model small. [Download](https://weilab.math.msu.edu/Downloads/TopoFormer/TopoFormer_pretrained_model.zip)

#### Finetuned models and performances
- Scoring


| Finetuned for scoring                                                | Training Set                  | Test Set| PCC | RMSE (kcal/mol) |
|-------------------------------------------------                     |-------------                  |---------|-    |-                |
| CASF-2007 [result](./Results)      | 1105                          | 195     |0.837|1.807|
| CASF-2007 small [result](./Results)| 1105                          | 195     |0.839|1.807|
| CASF-2013 [result](./Results)      | 2764                          | 195     |0.816|1.859|
| CASF-2016 [result](./Results)      | 3772                          | 285     |0.864|1.568|
| PDB v2016 [result](./Results)      | 3767                          | 290     |0.866|1.561|
| PDB v2020 [result](./Results)      | 18904 <br> (exclude core sets)|195<br>CASF-2007 core set|0.853|1.295|
|                                    |                               |195<br>CASF-2013 core set|0.832|1.301|
|                                    |                               |285<br>CASF-2016 core set|0.881|1.095|

Note, there are 20 TopoFormers are trained for each dataset with distinct random seeds to address initialization-related errors. And 20 gradient boosting regressor tree (GBRT) models are subsequently trained one these sequence-based features, which predictions can be found in the [results](./Results) folder. Then, 10 models were randomly selected from TopoFormer and GBDT models, respectively, the consensus predictions of these models was used as the final prediction result. The performance shown in the table is the average result of this process performed 400 times.

- Docking


| Finetuned for docking                                                | Success rate |
|-------------------------------------------------                     |-             |
| CASF-2007 [result](./Results)| 93.3%         |
| CASF-2013 [result](./Results)| 91.3%         |

- Screening

| Finetuned for screening                                              |Success rate on 1%|Success rate on 5%|Success rate on 10%|EF on 1%|EF on 5%|EF on 10%|
|-                                                                     | - | - | - | - | - | - |
| CASF-2013 |68%|81.5%|87.8%|29.6|9.7|5.6|

Note, the EF here means the enhancement factor. Each target protein has a finetuned model. [result](./Results) contains all predictions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code or the pre-trained models in your work, please cite our work. 
- Chen, Dong, Jian Liu, and Guo-Wei Wei. "Multiscale Topology-enabled Structure-to-Sequence Transformer for Protein-Ligand Interaction Predictions."

---

## Acknowledgements

This project has benefited from the use of the [Transformers](https://github.com/huggingface/transformers) library. Portions of the code in this project have been modified from the original code found in the Transformers repository.
