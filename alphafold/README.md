## AlphaFold Instructions 

### File Directory 
```
root
│   README.md
│   PPI_multitask_DMS_mega.py    
│
└───alphafold
│   │   preprocess.py
│   │   debug.py
│   │
│   └───cif (place zip files in this folder)
│   |   │   fold_1a4y.zip
│   |   │   fold_1a22.zip
│   |   │   ...
|   |
|   └───raw (unzip all zip files in cif to here)
│   |   └───fold_1a4y
│   |   └───fold_1a22
│   |   └───fold_1acb
│   |   └─── ...  
|   |
.   └───pdb (skempi2 fasta and pdb files)
.
.    
```

### Preprocessing 
```shell
cd ./alphafold/
python preprocess.py 
```
