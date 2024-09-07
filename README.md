# MT-TopLap <!-- [![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.12508&color=B31B1B)](https://www.google.com/) --> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Model Architecture

The multitask deep learning architecture of MT-TopLap model is shown below.
![Model Architecture](./MT-TopLap_dark.png#gh-dark-mode-only)
![Model Architecture](./MT-TopLap.png#gh-light-mode-only)

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
- Softwares to be installed for ./bin folder (See README at bin)

## Multi-task Pre-training Datasets

| Dataset                                                            | No. of samples | PDB ID                                                                       |
|--------------------------------------------------------------------|-------------------------------------|------------------------------------------------------------------------------|
| RBD-ACE2-1                                     | 3669                                | 6M0J                                                  |
| RBD-ACE2-2                                    | 1539                                | 6M0J                                                  |
| RBD-ACE2-3                            | 2223                                | 6M0J                                                  |
| RBD-CTC-455.2-1                             | 1539                                | 7KL9                                                    |
| RBD-CTC-455.2-2                            | 2831                                | 7KL9                                                    |
| BA.1-RBD-ACE2                            | 3800                                | 7T9L                                                   |
| BA.2-RBD-ACE2                                 | 3686                                | 7XB0                                                  |
| CAPRI  | 1862                                | 3R2X |
## 10-fold Cross-Validation Benchmarking 
| Dataset    | No. of Samples | No. of PPIs | PDB Source                      |
|------------|----------------|-------------|---------------------------------|
| Original   | 8338           | 319         | Download from RCSB              |
| AlphaFold3 | 8330           | 317         | Download from AlphaFold Server  |

## Downstream DMS Tasks
| Dataset         | No. of samples | PDB ID |
|-----------------|----------------|--------|
| RBD-hACE2       | 3649           | 6M0J   |
| RBD-cACE2       | 3625           | 7C8D   |
| RBD-bACE2       | 3646           | 7XA7   |
| RBD-dACE2       | 3487           | 8HG0   |
| BA.2-RBD-hACE2  | 3668           | 7ZF7   |
| BA.2-RBD-haACE2 | 3724           | 7YV8   |

The PDB files, mutation locations and mutation-induced binding free energy changes can be found in ./downstream/.

---
## Feature generation
### BLAST+ PSSM calculation
```shell
# Generate PSSM scoring matrix (Requires BLAST+ 2.10.1 and GCC 9.3.0)
python PPIprepare.py <PDB ID> <Partner A chains> <Partner B chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# examples
python PPIprepare.py 1A4Y A B A D 435 A 7.0 
```

### MIBPB calculation
Refer to https://weilab.math.msu.edu/MIBPB/ 

### Persistent Laplacian and Auxiliary Features 
```shell
# Generate persistent homology and auxiliary features
python PPIfeature.py <PDB ID> <Partner A chains> <Partner B chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>
python PPIfeature_Lap.py <PDB ID> <Partner A chains> <Partner B chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# examples
python PPIfeature.py 1A4Y A B A D 435 A 7.0
python PPIfeature_Lap.py 1A4Y A B A D 435 A 7.0 
```

### ESM Features 
```shell
# Generate transformer features
python PPIfeature_seq.py <PDB ID> <Partner A chains> <Partner B chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# examples
python PPIfeature_seq.py 1A4Y A B A D 435 A 7.0
```
The jobs folder contains the codes used to run feature generation process in a step-by-step procedure. 

---
## Validation Results
| Method                | R_p  | Description           | PDB Source |
|-----------------------|------|-----------------------|------------|
| MT-TopLap             | 0.88 | Freeze Last 3 Layers  | RCSB       |
| MT-TopLap<sup>E</sup>         | 0.88 | Freeze Even Layers    | RCSB       |
| MT-TopLap<sup>O</sup>         | 0.88 | Freeze Odd Layers     | RCSB       |
| MT-TopLap<sub>AF3</sub> | 0.86 | Freeze Last 3 Layers  | AlphaFold3 |
| MT-TopLap<sup>F</sup>         | 0.85 | Freeze First 3 Layers | RCSB       |

## Execute Model
```shell
usage: PPI_multitask_DMS_mega.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--momentum MOMENTUM]
                                 [--weight_decay WEIGHT_DECAY] [--no_cuda] [--seed SEED] [--log_interval LOG_INTERVAL]
                                 [--layers LAYERS] [--continue_train CONTINUE_TRAIN] [--prediction PREDICTION]
                                 [--pred PRED] [--cv CV] [--cv_type CV_TYPE] [--model MODEL]
                                 [--normalizer1_name NORMALIZER1_NAME] [--normalizer2_name NORMALIZER2_NAME]
                                 [--freeze FREEZE] [--skempi_pretrain SKEMPI_PRETRAIN] [--ft FT] [--finetune FINETUNE]
                                 [--debug DEBUG] [--DMS_data DMS_DATA] [--pred_DMS PRED_DMS]

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        input batch size for training (default: 50)
  --epochs EPOCHS       number of epochs to train (default: 500)
  --lr LR               learning rate (default: 0.001)
  --momentum MOMENTUM   SGD momentum (default: 0.9)
  --weight_decay WEIGHT_DECAY
                        SGD weight decay (default: 0)
  --no_cuda             disables CUDA training
  --seed SEED           random seed (default: 1)
  --log_interval LOG_INTERVAL
                        how many batches to wait before logging training status
  --layers LAYERS       neural network layers and neural numbers
  --continue_train CONTINUE_TRAIN
                        run training
  --prediction PREDICTION
                        prediction
  --pred PRED
  --cv CV               cv (launch validation test)
  --cv_type CV_TYPE     skempi2 or alphafold (select original PDB or AF3 validation)
  --model MODEL         prediction model
  --normalizer1_name NORMALIZER1_NAME
                        mega dataset
  --normalizer2_name NORMALIZER2_NAME
                        Lap and ESM dataset
  --freeze FREEZE       freeze weights and biases of hidden layers
  --skempi_pretrain SKEMPI_PRETRAIN
                        finetune DMS with SKEMPI2
  --ft FT               finetune pretrained model with DMS
  --finetune FINETUNE   finetune model with DMS data
  --debug DEBUG         debugging channel
  --DMS_data DMS_DATA   predict full DMS data
  --pred_DMS PRED_DMS   predict full DMS data
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---

## Citation
If you use this code in your work, please cite our work. 
- JunJie Wee, Guo-Wei Wei. Evaluation of AlphaFold 3’s Protein–Protein Complexes for Predicting Binding Free Energy Changes upon Mutation. Journal of Chemical Information and Modeling (2024). DOI: 10.1021/acs.jcim.4c00976
- JunJie Wee, Jiahui Chen, Guo-Wei Wei. Preventing future zoonosis: SARS-CoV-2 mutations enhance human–animal cross-transmission. Computers in Biology and Medicine 182 (2024). 109101. DOI: 10.1016/j.compbiomed.2024.109101
---
