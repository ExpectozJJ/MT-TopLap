import sys 
import os, zipfile
import numpy as np 
import glob 
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO

def extract(filelist):
    for name in filelist:
        if not os.path.exists(f'../raw/{name[:-4]}/'):
            os.mkdir(f'../raw/{name[:-4]}/')

        os.chdir(f'../raw/{name[:-4]}/')
        with zipfile.ZipFile(f'../../cif/{name}', 'r') as zip:
            zip.extractall()
        os.chdir(f'../../cif/')

def check(filelist):
    idx_list = open('../skempi2_list.txt', 'r')
    contents = idx_list.readlines()
    pdb_list = []
    for i in range(len(contents)):
        line = contents[i].split(',')
        #print(line)
        pdb_list.append(line[0])
    
    pdb_list = np.unique(pdb_list)
    ll = []
    for i in range(len(filelist)):
        ll.append(filelist[i][-8:-4].upper())
    print(ll)
    print(set(pdb_list)-set(ll))

os.chdir('./cif/')
filelist = glob.glob("fold_*.zip")
#print(filelist)

#check(filelist)
extract(filelist)
for name in filelist:
    os.chdir(f'../raw/{name[:-4]}/')
    for i in range(1):
        ciffile = f'{name[:-4]}_model_{i}.cif'
        pdbfile = f'{name[:-4]}_model_{i}.pdb'
        strucid = ciffile[:4] if len(ciffile)>4 else "1xxx"
        parser = MMCIFParser()
        structure = parser.get_structure(strucid, ciffile)
        s1 = structure.copy()

        io = PDBIO()
        io.set_structure(structure)
        io.save(pdbfile)

    os.chdir(f'../../cif/')
