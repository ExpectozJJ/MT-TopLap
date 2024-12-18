import numpy as np 
import os 
import re
from PPIstructure import get_structure
import time
import multiprocessing as mp

def feat_job(list_, folder1, folder2):
    numCPU = 10
    os.chdir("./7KL9_RBDb/")
    for ilist in list_:
        folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[4]}_{ilist[5]}_{ilist[6]}'
        dst = f'{folder1}/{folder_dir}'
        fp = open(f'{folder2}/{folder_dir}.job', 'w')
        fp.write('#!/bin/bash\n')
        fp.write('#SBATCH --time=08:00:00\n')
        fp.write('#SBATCH --nodes=1\n')
        fp.write('#SBATCH -A guowei-search\n')
        fp.write('#SBATCH --ntasks-per-node=1\n')
        fp.write('#SBATCH --cpus-per-task=1\n')
        fp.write('#SBATCH --constraint=\"amd20\"\n')
        fp.write('#SBATCH --mem=%dG\n'%(numCPU*4))
        fp.write(f'#SBATCH --job-name {folder_dir}\n')
        fp.write(f'#SBATCH --output=./feat_out/{folder_dir}.out\n')
        fp.write(f'cd /mnt/research/guowei-search.4/junjie-ppi/7KL9_RBDb/{dst}\n')
        #fp.write('source activate pytorch\n')
        fp.write('module purge\n')
        fp.write('ml -* GCC/7.3.0-2.30 dssp/3.1.4\n')
        fp.write('module load GCC/7.3.0-2.30 OpenMPI/3.1.1\n')
        fp.write('module load BLAST+/2.8.1\n')
        fp.write('python PPIfeature.py '+' '.join(ilist)+' 7.0\n')
        fp.write('python PPIfeature_Lap.py '+' '.join(ilist)+' 7.0\n')
        fp.close()
    os.chdir("../")
    return

def run_feat(list_, folder1, folder2):
    os.chdir("./7KL9_RBDb/")
    for i in range(len(list_)):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[4]}_{ilist[5]}_{ilist[6]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[3]}'

        opp_folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[6]}_{ilist[5]}_{ilist[4]}'
        if not os.path.exists(f'{dst}/{folder_dir}_FRI.npy') or not os.path.exists(f'{dst}/{folder_dir}_PH0.npy') \
            or not os.path.exists(f'{dst}/{folder_dir}_PH12.npy')  \
            or not os.path.exists(f'{dst}/{opp_folder_dir}_FRI.npy') or not os.path.exists(f'{dst}/{opp_folder_dir}_PH0.npy') \
            or not os.path.exists(f'{dst}/{opp_folder_dir}_PH12.npy') \
            or not os.path.exists(f'{dst}/{folder_dir}_Lap_b.npy') or not os.path.exists(f'{dst}/{opp_folder_dir}_Lap_b.npy'):
            #or os.stat(f'{dst}/{folder_dir}_Lap_b.npy').st_size == 0 or os.stat(f'{dst}/{opp_folder_dir}_Lap_b.npy').st_size == 0:
        #tmp = np.load(f'{dst}/{folder_dir}_Lap_b.npy', allow_pickle=True)
        #if np.shape(tmp)[0] != 1296:
            os.system(f'sbatch {folder2}/{folder_dir}.job')
    os.chdir("../")
    return 

def create_blastjob(list_, folder1, folder2):
    numCPU = 10
    os.chdir("./7KL9_RBDb/")
    for ilist in list_:
        folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[4]}_{ilist[5]}_{ilist[6]}'
        dst = f'{folder1}/{folder_dir}'
        fp = open(f'{folder2}/{folder_dir}.job', 'w')
        fp.write('#!/bin/bash\n')
        fp.write('#SBATCH --time=08:00:00\n')
        fp.write('#SBATCH --nodes=1\n')
        fp.write('#SBATCH -A guowei-search\n')
        fp.write('#SBATCH --ntasks-per-node=1\n')
        fp.write('#SBATCH --cpus-per-task=1\n')
        fp.write('#SBATCH --constraint=\"amd20\"\n')
        fp.write('#SBATCH --mem=%dG\n'%(numCPU*4))
        fp.write(f'#SBATCH --job-name {folder_dir}\n')
        fp.write(f'#SBATCH --output=./blast_out/{folder_dir}.out\n')
        fp.write(f'cd /mnt/research/guowei-search.4/junjie-ppi/7KL9_RBDb/{dst}\n')
        #fp.write('source activate pytorch\n')
        fp.write('module purge\n')
        fp.write('module load GCC/9.3.0 OpenMPI/4.0.3\n')
        fp.write('module load BLAST+/2.10.1\n')
        fp.write('python PPIprepare.py '+' '.join(ilist)+' 7.0\n')
        #fp.write('python PPIfeature_Lap.py '+' '.join(ilist[:-1])+'\n')
        fp.close()
    os.chdir("../")
    return

def blast_jobs(list_, folder1, folder2):
    os.chdir("./7KL9_RBDb/")
    for i in range(len(list_)):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[4]}_{ilist[5]}_{ilist[6]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[3]}'
        if not os.path.exists(f'{dst}/{filename}_WT.pssm') or not os.path.exists(f'{dst}/{filename}_MT.pssm'):
            os.system(f'sbatch {folder2}/{folder_dir}.job')
    os.chdir("../")
    return 

def seq_job(list_, folder1, folder2):
    numCPU = 10
    os.chdir("./7KL9_RBDb/")
    for ilist in list_:
        folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[4]}_{ilist[5]}_{ilist[6]}'
        dst = f'{folder1}/{folder_dir}'
        fp = open(f'{folder2}/{folder_dir}.job', 'w')
        fp.write('#!/bin/bash\n')
        fp.write('#SBATCH --time=03:00:00\n')
        fp.write('#SBATCH --nodes=1\n')
        fp.write('#SBATCH -A guowei-search\n')
        fp.write('#SBATCH --ntasks-per-node=1\n')
        fp.write('#SBATCH --cpus-per-task=1\n')
        fp.write('#SBATCH --constraint=\"amd20\"\n')
        fp.write('#SBATCH --mem=%dG\n'%(numCPU*10))
        fp.write(f'#SBATCH --job-name {folder_dir}\n')
        fp.write(f'#SBATCH --output=./seq_out/{folder_dir}.out\n')
        fp.write(f'cd /mnt/research/guowei-search.4/junjie-ppi/7KL9_RBDb/{dst}\n')
        #fp.write('source activate pytorch\n')
        fp.write('python PPIfeature_seq.py '+' '.join(ilist)+' 7.0\n')
        fp.close()
    os.chdir("../")
    return

def run_seq(list_, folder1, folder2):
    os.chdir("./7KL9_RBDb/")
    for i in range(len(list_)):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[4]}_{ilist[5]}_{ilist[6]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[3]}'

        opp_folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[6]}_{ilist[5]}_{ilist[4]}'
        if not os.path.exists(f'{dst}/{folder_dir}_seq.npy') or not os.path.exists(f'{dst}/{opp_folder_dir}_seq.npy'):
            os.system(f'sbatch {folder2}/{folder_dir}.job')
    os.chdir("../")
    return 

def gen_PDBs(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT):
    #PDBid, Antibody, Antigen, Chain, resWT, resID, resMT, pH = ilist[0], ilist[1], ilist[2], ilist[3], ilist[4], ilist[5], ilist[6], ilist[7]
    print(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT)
    if not os.path.exists("features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT)):
        os.system("mkdir features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT))
    
    curr_dir = "features/{}_{}_{}_{}_{}/".format(PDBid, Chain, resWT, resID, resMT)
    os.chdir("features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT))
    #os.system("ln -s 7KL9_RBDb/{}.pdb {}{}.pdb".format(PDBid, curr_dir, PDBid))
    os.system("ln -s ../../../code/TopLapNet/bin/jackal.dir jackal.dir")
    os.system("ln -s ../../../code/TopLapNet/bin/profix profix")
    os.system("ln -s ../../../code/TopLapNet/bin/scap scap")
    os.system("ln -s ../../../code/TopLapNet/code/PPIstructure.py PPIstructure.py")
    os.system("ln -s ../../../code/TopLapNet/code/PPIprotein.py PPIprotein.py")
    os.system("ln -s ../../../code/TopLapNet/code/PPIfeature.py PPIfeature.py")
    os.system("ln -s ../../../code/TopLapNet/code/PPIfeature_Lap.py PPIfeature_Lap.py")
    os.system("ln -s ../../../code/TopLapNet/code/PPIfeature_seq.py PPIfeature_seq.py")
    os.system("ln -s ../../../code/TopLapNet/code/PPIcomplex.py PPIcomplex.py")
    os.system("ln -s ../../../code/TopLapNet/code/PPIprepare.py PPIprepare.py")
    os.system("ln -s ../../../code/TopLapNet/code/src src")

    if not os.path.exists(PDBid+'.pdb'):
        os.system('wget https://files.rcsb.org/download/'+PDBid+'.pdb')

    s = get_structure(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT, pH='7.0')
    s.generateMutedPartnerPDBs()
    s.generateMutedPartnerPQRs()
    s.generateComplexPDBs()
    s.readFASTA()
    s.writeFASTA()
    #filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
    #if not os.path.exists(PDBid+'_'+Chain+'.englist'):
    #print(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT)
    #os.system("python PPIprepare.py {} {} {} {} {} {} {} 7.0".format(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT))
    #os.system("python PPIfeature_Lap.py {} {} {} {} {} {} {} 7.0".format(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT))
    os.chdir("../../")

def mp_gen_PDBs(seq_list):

    os.chdir("./7KL9_RBDb/")
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    results = p.starmap(gen_PDBs, seq_list)
    p.close()
    p.join()
    os.chdir("../")

def check_pssm(list_, folder1, folder2):
    os.chdir("./7KL9_RBDb/")
    for i in range(len(list_)):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[3]}_{ilist[4]}_{ilist[5]}_{ilist[6]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[3]}'
        #if not os.path.exists(f'{dst}/{filename}_WT.pssm') or not os.path.exists(f'{dst}/{filename}_MT.pssm'):
        file1 = open(f'{dst}/{filename}_WT.pssm')
        c1 = file1.readlines()
        file2 = open(f'{dst}/{filename}_MT.pssm')
        c2 = file2.readlines()
        if c1[-1][:3]!="PSI" or c2[-1][:3]!="PSI":
            print(folder_dir)
            os.system(f'sbatch {folder2}/{folder_dir}.job')
    os.chdir("../")
    return 

def dataset_list(filename):
    dataset = []
    fp = open(filename)
    for line in fp:
        line_split = re.split(',|\n', line)
        dataset.append(line_split[:-2])
    fp.close()
    return dataset

if not os.path.exists("./7KL9_RBDb/features/"):
    os.system("mkdir ./7KL9_RBDb/features/")

seq_list = dataset_list("./7KL9_RBDb/7KL9_RBDb.txt")
#print(seq_list)
#mp_gen_PDBs(seq_list)

if not os.path.exists("./7KL9_RBDb/blast_jobs/"):
    os.mkdir("./7KL9_RBDb/blast_jobs/")
if not os.path.exists("./7KL9_RBDb/blast_out/"):
    os.mkdir("./7KL9_RBDb/blast_out/")

create_blastjob(seq_list, "features", "blast_jobs")
#blast_jobs(seq_list, "features", "blast_jobs")
#check_pssm(seq_list, "features", "blast_jobs")

if not os.path.exists("./7KL9_RBDb/feat_jobs/"):
    os.mkdir("./7KL9_RBDb/feat_jobs/")

if not os.path.exists("./7KL9_RBDb/feat_out/"):
    os.mkdir("./7KL9_RBDb/feat_out/")

feat_job(seq_list, "features", "feat_jobs")
#run_feat(seq_list, "features", "feat_jobs")

if not os.path.exists("./7KL9_RBDb/seq_jobs/"):
    os.mkdir("./7KL9_RBDb/seq_jobs/")

if not os.path.exists("./7KL9_RBDb/seq_out/"):
    os.mkdir("./7KL9_RBDb/seq_out/")

seq_job(seq_list, "features", "seq_jobs")
run_seq(seq_list, "features", "seq_jobs")

