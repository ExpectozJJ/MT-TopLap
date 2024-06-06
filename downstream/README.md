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

### Installation

```
git clone https://github.com/ExpectozJJ/MT-TopLap
```

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
### BLAST Features
```shell
# Generate PSSM scoring matrix (Requires BLAST+ 2.10.1 and GCC 9.3.0)
python PPIprepare.py <PDB ID> <Partner A chains> <Partner B chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# examples
python PPIprepare.py 1A4Y A B A D 435 A 7.0 
```

### MIBPB Features 
Refer to https://weilab.math.msu.edu/MIBPB/ 

### Topological and auxiliary Features 
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

---
## Validation Results

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

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---

## Citation
If you use this code in your work, please cite our work. 
- JunJie Wee and Guo-Wei Wei. "Benchmarking AlphaFold3's protein-protein complex accuracy and machine learning prediction reliability for binding free energy changes upon mutation."
- JunJie Wee, Jiahui Chen and Guo-Wei Wei. "Preventing future zoonosis: SARS-CoV-2 mutations enhancing human-animal cross-transmission."
---
