import argparse, sys, time, random, torch, os, joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import scipy
from Bio import *
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def dataset_list(filename):
    dataset = []
    fp = open(filename)
    for line in fp:
        line_split = re.split(',|\n', line)
        dataset.append(line_split[:-2])
    fp.close()
    return dataset

class scaler():
    def __init__(self, X):
        self.scaler = StandardScaler().fit(X)

    def fit(self, X_in):
        return self.scaler.transform(X_in)

    def save_normalizer(self, name):
        if not os.path.exists('model/'): os.system('mkdir model/')
        joblib.dump(self.scaler, f'model/normalizer{name}.pkl')
        print(f'normalizer saved as normalizer{name}.pkl')

def RMSE(ypred, yexact):
    return torch.sqrt(torch.sum((ypred-yexact)**2)/ypred.shape[0])

def PCC(ypred, yexact):
    a = ypred.cpu().numpy().ravel()
    b = yexact.cpu().numpy().ravel()
    pcc = stats.pearsonr(a, b)
    return pcc

class TopLapNet(Dataset):
    def __init__(self, X, y, transforms=transforms.Compose([])):
        self.X = X
        self.labels = y
        self.transforms = transforms

    def __getitem__(self, index):
        X_array2tensor = torch.from_numpy(self.X[index]).float()
        if self.transforms is not None:
            X_array2tensor = self.transforms(X_array2tensor)
        return (X_array2tensor, self.labels[index])

    def __len__(self):
        return self.X.shape[0]

class MultitaskModule(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultitaskModule, self).__init__()

        # input layer and initialize weights
        self.input_layer = nn.Linear(D_in, H[0], bias=True)
        nn.init.xavier_uniform_(self.input_layer.weight)

        # hiden layer and initialize weights
        self.hiden_layers = nn.ModuleList([nn.Linear(H[i], H[i+1], bias=True) 
                                           for i in range(len(H)-1)])
        for hiden_layer in self.hiden_layers:
            nn.init.xavier_uniform_(hiden_layer.weight)

        # output layer and initialize weights
        self.output_layer = nn.Linear(H[-1], D_out, bias=True)
        nn.init.xavier_uniform_(self.output_layer.weight)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)

    def add_channel(self, H, D_out):
        curr = self.output_layer.weight.data
        curr_bias = self.output_layer.bias.data
        print(self.output_layer.bias)
        #print(self.output_layer.weight.data)
        hl_input = torch.zeros([1, H[-1]])
        nn.init.xavier_uniform_(hl_input)
        new_weights = torch.cat([curr, hl_input], dim=0)
        self.output_layer = nn.Linear(H[-1], D_out+1, bias=True)
        self.output_layer.weight.data = torch.tensor(new_weights)
        self.output_layer.bias.data[:D_out] = curr_bias
        print(self.output_layer.bias)
        #print(self.output_layer.weight.data)

    def forward(self, X):
        X = F.relu(self.input_layer(X))
        X = self.dropout(X)
        for hiden_layer in self.hiden_layers:
            X = F.relu(hiden_layer(X))
            X = self.dropout(X)
        y = self.output_layer(X)
        return y

def train(model, device, train_loader, criterion, optimizer, task_idx):
    model.train() # tells your model that you are training the model
    for (data, target) in train_loader:
        # move tensor to computing device ('gpu' or 'cpu')
        data, target = data.to(device), target.to(device).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data)[:, task_idx].view(-1, 1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    return

def test(model, device, test_loader, epoch, task_idx):
    #print(task_idx)
    model.eval() # tell that you are testing, == model.train(model=False)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)[:, task_idx].view(-1, 1)
            test_loss = F.mse_loss(output, target, reduction='sum').item()
            test_loss /= len(test_loader.dataset)
            #print(np.isnan(output.cpu().numpy()).any())
            #print(np.isnan(target.cpu().numpy()).any())
            pcc = PCC(output, target)[0]
            rmse = RMSE(output, target)
            print('Epoch: %s, test_loss: %.4f, RMSE: %.4f, PCC: %.4f'%(epoch, test_loss, rmse, pcc))
            return output.cpu().numpy(), target.cpu().numpy()

def randomize_features(X_val, y_val):
    num_sample, num_feature = X_val.shape
    index = np.arange(num_sample)
    np.random.shuffle(index)
    X_val = X_val[index, :]
    y_val = y_val[index]
    return X_val, y_val

def load_data(args):
    normalizer1 = joblib.load(f'model/normalizer{args.normalizer1_name}.pkl')
    normalizer2 = joblib.load(f'model/normalizer{args.normalizer2_name}.pkl')

    # log2 enrichment and BFE features and labels
    X_val_3R2X = np.load('./3R2X/X_val_3R2X.npy')
    X_val_4EEF = np.load('./4EEF/X_val_4EEF.npy')
    X_val_CAPRI = np.concatenate((X_val_3R2X, X_val_4EEF), axis=0)

    X_val_6M0J_ACE2 = np.load('./6M0J_ACE2/X_val_6M0J_ACE2.npy')
    X_val_6M0J_RBDc = np.load('./6M0J_RBDc/X_val_6M0J_RBDc.npy')
    X_val_6M0J_RBDh = np.load('./6M0J_RBDh/X_val_6M0J_RBDh.npy')
    X_val_7KL9_RBDb = np.load('./7KL9_RBDb/X_val_7KL9_RBDb.npy')
    X_val_7KL9_CTCa = np.load('./7KL9_CTCa/X_val_7KL9_CTCa.npy')
    X_val_7XB0_RBD = np.load('./7XB0_RBD/X_val_7XB0_RBD.npy')
    X_val_7T9L_RBD = np.load('./7T9L_RBD/X_val_7T9L_RBD.npy')
    #X_val_skempi2 = np.load('./skempi2/X_S8338.npy')

    # log2 enrichment and BFE features and labels (Lap and ESM features)
    X_val_3R2X_Lap_ESM = np.load('./3R2X/X_val_3R2X_Lap_ESM.npy')
    X_val_4EEF_Lap_ESM = np.load('./4EEF/X_val_4EEF_Lap_ESM.npy')
    X_val_CAPRI_Lap_ESM = np.concatenate((X_val_3R2X_Lap_ESM, X_val_4EEF_Lap_ESM), axis=0)

    X_val_6M0J_ACE2_Lap_ESM = np.load('./6M0J_ACE2/X_val_6M0J_ACE2_Lap_ESM.npy')
    X_val_6M0J_RBDc_Lap_ESM = np.load('./6M0J_RBDc/X_val_6M0J_RBDc_Lap_ESM.npy')
    X_val_6M0J_RBDh_Lap_ESM = np.load('./6M0J_RBDh/X_val_6M0J_RBDh_Lap_ESM.npy')
    X_val_7KL9_RBDb_Lap_ESM = np.load('./7KL9_RBDb/X_val_7KL9_RBDb_Lap_ESM.npy')
    X_val_7KL9_CTCa_Lap_ESM = np.load('./7KL9_CTCa/X_val_7KL9_CTCa_Lap_ESM.npy')
    X_val_7XB0_RBD_Lap_ESM = np.load('./7XB0_RBD/X_val_7XB0_RBD_Lap_ESM.npy')
    X_val_7T9L_RBD_Lap_ESM = np.load('./7T9L_RBD/X_val_7T9L_RBD_Lap_ESM.npy')
    #X_val_skempi2_Lap_ESM = np.load('./skempi2/X_S8338_Lap_ESM.npy')

    y_val_3R2X = np.load('./3R2X/Y_3R2X.npy').reshape((-1, 1))
    y_val_4EEF = np.load('./4EEF/Y_4EEF.npy').reshape((-1, 1))
    y_val_CAPRI = np.concatenate((y_val_3R2X, y_val_4EEF), axis=0)

    y_val_6M0J_ACE2 = np.load('./6M0J_ACE2/Y_6M0J_ACE2.npy').reshape((-1, 1))
    y_val_6M0J_RBDc = np.load('./6M0J_RBDc/Y_6M0J_RBDc.npy').reshape((-1, 1))
    y_val_6M0J_RBDh = np.load('./6M0J_RBDh/Y_6M0J_RBDh.npy').reshape((-1, 1))
    y_val_7KL9_RBDb = np.load('./7KL9_RBDb/Y_7KL9_RBDb.npy').reshape((-1, 1))
    y_val_7KL9_CTCa = np.load('./7KL9_CTCa/Y_7KL9_CTCa.npy').reshape((-1, 1))
    y_val_7XB0_RBD = np.load('./7XB0_RBD/Y_7XB0_RBD.npy').reshape((-1, 1))
    y_val_7T9L_RBD = np.load('./7T9L_RBD/Y_7T9L_RBD.npy').reshape((-1, 1))
    #y_val_skempi2 = np.load('./skempi2/Y_S8338.npy').reshape((-1, 1))

    X_val_6M0J_ACE2 = normalizer1.transform(X_val_6M0J_ACE2)
    X_val_6M0J_RBDc = normalizer1.transform(X_val_6M0J_RBDc)
    X_val_6M0J_RBDh = normalizer1.transform(X_val_6M0J_RBDh)
    X_val_7KL9_CTCa = normalizer1.transform(X_val_7KL9_CTCa)
    X_val_7KL9_RBDb = normalizer1.transform(X_val_7KL9_RBDb)
    X_val_7T9L_RBD = normalizer1.transform(X_val_7T9L_RBD)
    X_val_7XB0_RBD = normalizer1.transform(X_val_7XB0_RBD)
    X_val_CAPRI = normalizer1.transform(X_val_CAPRI)

    X_val_6M0J_ACE2_Lap_ESM = normalizer2.transform(X_val_6M0J_ACE2_Lap_ESM)
    X_val_6M0J_RBDc_Lap_ESM = normalizer2.transform(X_val_6M0J_RBDc_Lap_ESM)
    X_val_6M0J_RBDh_Lap_ESM = normalizer2.transform(X_val_6M0J_RBDh_Lap_ESM)
    X_val_7KL9_CTCa_Lap_ESM = normalizer2.transform(X_val_7KL9_CTCa_Lap_ESM)
    X_val_7KL9_RBDb_Lap_ESM = normalizer2.transform(X_val_7KL9_RBDb_Lap_ESM)
    X_val_7T9L_RBD_Lap_ESM = normalizer2.transform(X_val_7T9L_RBD_Lap_ESM)
    X_val_7XB0_RBD_Lap_ESM = normalizer2.transform(X_val_7XB0_RBD_Lap_ESM)
    X_val_CAPRI_Lap_ESM = normalizer2.transform(X_val_CAPRI_Lap_ESM)

    X_val_6M0J_ACE2 = np.concatenate((X_val_6M0J_ACE2, X_val_6M0J_ACE2_Lap_ESM), axis=1)
    X_val_6M0J_RBDc = np.concatenate((X_val_6M0J_RBDc, X_val_6M0J_RBDc_Lap_ESM), axis=1)
    X_val_6M0J_RBDh = np.concatenate((X_val_6M0J_RBDh, X_val_6M0J_RBDh_Lap_ESM), axis=1)
    X_val_7KL9_CTCa = np.concatenate((X_val_7KL9_CTCa, X_val_7KL9_CTCa_Lap_ESM), axis=1)
    X_val_7KL9_RBDb = np.concatenate((X_val_7KL9_RBDb, X_val_7KL9_RBDb_Lap_ESM), axis=1)
    X_val_7T9L_RBD = np.concatenate((X_val_7T9L_RBD, X_val_7T9L_RBD_Lap_ESM), axis=1)
    X_val_7XB0_RBD = np.concatenate((X_val_7XB0_RBD, X_val_7XB0_RBD_Lap_ESM), axis=1)
    X_val_CAPRI = np.concatenate((X_val_CAPRI, X_val_CAPRI_Lap_ESM), axis=1)

    X_val_6M0J_ACE2, y_val_6M0J_ACE2 = randomize_features(X_val_6M0J_ACE2, y_val_6M0J_ACE2)
    X_val_6M0J_RBDc, y_val_6M0J_RBDc = randomize_features(X_val_6M0J_RBDc, y_val_6M0J_RBDc)
    X_val_6M0J_RBDh, y_val_6M0J_RBDh = randomize_features(X_val_6M0J_RBDh, y_val_6M0J_RBDh)
    X_val_7KL9_CTCa, y_val_7KL9_CTCa = randomize_features(X_val_7KL9_CTCa, y_val_7KL9_CTCa)
    X_val_7KL9_RBDb, y_val_7KL9_RBDb = randomize_features(X_val_7KL9_RBDb, y_val_7KL9_RBDb)
    X_val_7T9L_RBD,  y_val_7T9L_RBD  = randomize_features(X_val_7T9L_RBD,  y_val_7T9L_RBD)
    X_val_7XB0_RBD,  y_val_7XB0_RBD  = randomize_features(X_val_7XB0_RBD,  y_val_7XB0_RBD)
    #X_val_skempi2,   y_val_skempi2   = randomize_features(X_val_skempi2,   y_val_skempi2)
    X_val_CAPRI,     y_val_CAPRI     = randomize_features(X_val_CAPRI,     y_val_CAPRI)

    # Building train set
    X_val = []
    y_val = []

    #X_val.append(normalizer.transform(X_val_skempi2))
    #y_val.append(y_val_skempi2)

    #if args.leave_who != '6M0J_ACE2':
    X_val.append(X_val_6M0J_ACE2)
    y_val.append(y_val_6M0J_ACE2)

    #if args.leave_who != '6M0J_RBDc':
    X_val.append(X_val_6M0J_RBDc)
    y_val.append(y_val_6M0J_RBDc)

    #if args.leave_who != '6M0J_RBDh':
    X_val.append(X_val_6M0J_RBDh)
    y_val.append(y_val_6M0J_RBDh)

    #if args.leave_who != 'CAPRI':
    X_val.append(X_val_CAPRI)
    y_val.append(y_val_CAPRI)

    #if args.leave_who != '7KL9_CTCa':
    X_val.append(X_val_7KL9_CTCa)
    y_val.append(y_val_7KL9_CTCa)

    #if args.leave_who != '7KL9_RBDb':
    X_val.append(X_val_7KL9_RBDb)
    y_val.append(y_val_7KL9_RBDb)

    #if args.leave_who != '7T9L_RBD':
    X_val.append(X_val_7T9L_RBD)
    y_val.append(y_val_7T9L_RBD)

    #if args.leave_who != '7XB0_RBD':
    X_val.append(X_val_7XB0_RBD)
    y_val.append(y_val_7XB0_RBD)

    return X_val, y_val

def generate_model(args):
    tic = time.perf_counter()
    print(args)
    torch.manual_seed(args.seed)

    # normalization
    #normalize_features(args)

    # load data
    X_val, y_val = load_data(args)
    num_channel = len(X_val)+1

    # setup device cuda or cpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # setup dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # num_workers=0?
    train_loader = []
    test_loader = []
    for i in range(1, num_channel):
        X_train = TopLapNet(X_val[i-1], y_val[i-1])
        train_loader.append(DataLoader(dataset=X_train, \
                                       batch_size=args.batch_size, \
                                       shuffle=True, \
                                       **kwargs))
        test_loader.append(DataLoader(dataset=X_train, \
                                      batch_size=len(X_train), \
                                      shuffle=True, \
                                      **kwargs))
 
    hiden_layer = [int(i) for i in args.layers.split(',')]
    if args.continue_train:
        print('loading multitask_C%d model<<<<<<<<<<<<<<<<<<'%num_channel)
        model = MultitaskModule(X_val[0].shape[1], hiden_layer, num_channel)
        model.load_state_dict(torch.load(f'model/multitask_C{num_channel}_pretrained_DMS_alphafold.pkl'))
        model = model.to(device)
    else:
        model = MultitaskModule(X_val[0].shape[1], hiden_layer, num_channel).to(device)
        if not os.path.exists('model/'):
            os.system('mkdir model/')
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)
    lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    for epoch in range(args.epochs):
        for task_idx in range(1, num_channel):
            train(model, device, train_loader[task_idx-1], criterion, optimizer, task_idx)
            if epoch%args.log_interval == 0:
                if task_idx==1: print('epoch %d >>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
                print(f'test data shape {X_val[task_idx-1].shape}')
                test(model, device, test_loader[task_idx-1], str(epoch), task_idx)
                if task_idx==num_channel-1 and epoch!=0:
                    print(f'epoch {epoch}, model saved multitask_C{num_channel}_pretrained_DMS_alphafold')
                    torch.save(model.state_dict(), f'model/multitask_C{num_channel}_pretrained_DMS_alphafold.pkl')
                    #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_{args.leave_who}_pred.pkl')
        lr_adjust.step()
    for task_idx in range(1, num_channel):
        print(f'test data shape {X_val[task_idx-1].shape}')
        test(model, device, test_loader[task_idx-1], 'End', task_idx)
        print(f'epoch {epoch}, model saved multitask_C{num_channel}_pretrained_DMS_alphafold')
        torch.save(model.state_dict(), f'model/multitask_C{num_channel}_pretrained_DMS_alphafold.pkl')
        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_{args.leave_who}_pred.pkl')
    toc = time.perf_counter()
    print('Elapsed time: %.1f [min]'%((toc-tic)/60))
    return

def alphafold_cv(args):
    tic = time.perf_counter()

    # setup device cuda or cpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # setup dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # num_workers=0?

    num_channel = 9

    X_val = np.load('./alphafold/X_val_alphafold.npy')
    normalizer1 = joblib.load('model/normalizer_mega_alphafold.pkl')
    X_val = normalizer1.transform(X_val)

    X_val_Lap_ESM = np.load('./alphafold/X_val_alphafold_Lap_ESM.npy')
    normalizer2 = joblib.load('model/normalizer_Lap_ESM_alphafold.pkl')
    X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

    X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)

    y_val = np.load('./alphafold/Y_alphafold.npy').reshape((-1, 1))
    print('The data shape', X_val.shape, ', label size', y_val.shape)

    #X_val, y_val = randomize_features(X_val, y_val)
    X_dms, y_dms = load_data(args)

    y_pred = np.zeros(len(y_val))
    y_real = np.zeros(len(y_val))
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(X_val)):
        # setup dataloader
        X_train, X_test = X_val[train_idx], X_val[test_idx]
        y_train, y_test = y_val[train_idx], y_val[test_idx]
        train_dataset = TopLapNet(X_train, y_train)
        test_dataset  = TopLapNet(X_test, y_test)
        valid_loader = []
        valid_loader.append(DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs))
        for i in range(1, num_channel):
            X_tmp = TopLapNet(X_dms[i-1], y_dms[i-1])
            valid_loader.append(DataLoader(dataset=X_tmp, batch_size=args.batch_size, **kwargs))
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs)
        test_loader  = DataLoader(dataset=test_dataset, batch_size=len(test_idx), **kwargs)
        
        input_layer = X_val.shape[1]
        num_channel = 9
        #num_channel = 8
        hiden_layer = [int(i) for i in args.layers.split(',')]
        model = MultitaskModule(input_layer, hiden_layer, num_channel)
        #print(model.hiden_layers)

        print('loading multitask_C%d model<<<<<<<<<<<<<<<<<<'%num_channel)
        model.load_state_dict(torch.load(f'model/multitask_C{num_channel}_pretrained_DMS_alphafold.pkl'))
        freeze_args = [bool(int(i)) for i in args.freeze.split(',')]
        model.input_layer.requires_grad_(freeze_args[0])
        for i in range(1, len(freeze_args)):
            model.hiden_layers[i-1].requires_grad_(freeze_args[i])
        model = model.to(device)

        criterion = nn.MSELoss()
        #optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
        lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
        print(f'test data shape {X_test.shape}')
        for epoch in range(args.epochs):
            #for task_idx in range(num_channel):
            train(model, device, train_loader, criterion, optimizer, 0)
            for task_idx in range(num_channel):
                if epoch%args.log_interval == 0:
                    if task_idx==0: print('epoch %d >>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
                    test(model, device, valid_loader[task_idx], str(epoch), task_idx)
                    #print(f'test data shape {X_test.shape}')
                    #if task_idx==num_channel-1 and epoch!=0:
                        #print(f'epoch {epoch}, model saved multitask_C{num_channel}_pretrained_DMS')
                        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_pretrained_DMS.pkl')
                        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_{args.leave_who}_pred.pkl')
            if epoch%args.log_interval == 0:
                print('Test Set >>>>>>>>>>>>>>>>>>>>>>>>')
                test(model, device, test_loader, str(epoch), 0)
            lr_adjust.step()

        #print(f'test data shape {X_test.shape}')
        y_pred_i, y_real_i = test(model, device, test_loader, 'End', 0)
        y_pred[test_idx] = np.reshape(y_pred_i, len(y_pred_i))
        y_real[test_idx] = np.reshape(y_real_i, len(y_real_i))
        #print(f'epoch {epoch}, model saved multitask_C{num_channel}_pretrained_DMS')
        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_pretrained_DMS.pkl')
        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_{args.leave_who}_pred.pkl')

    fp = open(f'./alphafold/s8330_cv.txt', 'w')
    for i in range(len(y_real)):
        fp.write(f'{y_pred[i]} {y_real[i]}\n')
    fp.close()
    pcc = stats.pearsonr(y_pred, y_real)[0]
    rmse = np.sqrt(mean_squared_error(y_pred, y_real))
    toc = time.perf_counter()
    print('Final RMSE: %.3f, Rp: %.4f\nElapsed time: %.1f [min]'%(rmse, pcc, (toc-tic)/60))

def skempi_cv(args):
    tic = time.perf_counter()

    # setup device cuda or cpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # setup dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # num_workers=0?

    num_channel = 9

    X_val = np.load('./skempi2/X_val_S8338.npy')
    normalizer1 = joblib.load('model/normalizer_mega.pkl')
    X_val = normalizer1.transform(X_val)

    X_val_Lap_ESM = np.load('./skempi2/X_val_S8338_Lap_ESM.npy')
    normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
    X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

    X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)[::2]

    y_val = np.load('./skempi2/Y_S8338.npy').reshape((-1, 1))[::2]
    print('The data shape', X_val.shape, ', label size', y_val.shape)

    X_val, y_val = randomize_features(X_val, y_val)
    X_dms, y_dms = load_data(args)

    y_pred = np.zeros(len(y_val))
    y_real = np.zeros(len(y_val))
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(X_val)):
        # setup dataloader
        X_train, X_test = X_val[train_idx], X_val[test_idx]
        y_train, y_test = y_val[train_idx], y_val[test_idx]
        train_dataset = TopLapNet(X_train, y_train)
        test_dataset  = TopLapNet(X_test, y_test)
        valid_loader = []
        valid_loader.append(DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs))
        for i in range(1, num_channel):
            X_tmp = TopLapNet(X_dms[i-1], y_dms[i-1])
            valid_loader.append(DataLoader(dataset=X_tmp, batch_size=args.batch_size, **kwargs))
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs)
        test_loader  = DataLoader(dataset=test_dataset, batch_size=len(test_idx), **kwargs)
        
        input_layer = X_val.shape[1]
        num_channel = 9
        #num_channel = 8
        hiden_layer = [int(i) for i in args.layers.split(',')]
        model = MultitaskModule(input_layer, hiden_layer, num_channel)
        #print(model.hiden_layers)

        print('loading multitask_C%d model<<<<<<<<<<<<<<<<<<'%num_channel)
        model.load_state_dict(torch.load(f'model/multitask_C{num_channel}_pretrained_DMS.pkl'))
        freeze_args = [bool(int(i)) for i in args.freeze.split(',')]
        model.input_layer.requires_grad_(freeze_args[0])
        for i in range(1, len(freeze_args)):
            model.hiden_layers[i-1].requires_grad_(freeze_args[i])
        model = model.to(device)

        criterion = nn.MSELoss()
        #optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
        lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
        print(f'test data shape {X_test.shape}')
        for epoch in range(args.epochs):
            #for task_idx in range(num_channel):
            train(model, device, train_loader, criterion, optimizer, 0)
            for task_idx in range(num_channel):
                if epoch%args.log_interval == 0:
                    if task_idx==0: print('epoch %d >>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
                    test(model, device, valid_loader[task_idx], str(epoch), task_idx)
                    #print(f'test data shape {X_test.shape}')
                    #if task_idx==num_channel-1 and epoch!=0:
                        #print(f'epoch {epoch}, model saved multitask_C{num_channel}_pretrained_DMS')
                        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_pretrained_DMS.pkl')
                        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_{args.leave_who}_pred.pkl')
            if epoch%args.log_interval == 0:
                print('Test Set >>>>>>>>>>>>>>>>>>>>>>>>')
                test(model, device, test_loader, str(epoch), 0)
            lr_adjust.step()

        #print(f'test data shape {X_test.shape}')
        y_pred_i, y_real_i = test(model, device, test_loader, 'End', 0)
        y_pred[test_idx] = np.reshape(y_pred_i, len(y_pred_i))
        y_real[test_idx] = np.reshape(y_real_i, len(y_real_i))
        #print(f'epoch {epoch}, model saved multitask_C{num_channel}_pretrained_DMS')
        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_pretrained_DMS.pkl')
        #torch.save(model.state_dict(), f'model/multitask_C{num_channel}_{args.leave_who}_pred.pkl')

    fp = open(f'./skempi2/s4169_cv.txt', 'w')
    for i in range(len(y_real)):
        fp.write(f'{y_pred[i]} {y_real[i]}\n')
    fp.close()
    pcc = stats.pearsonr(y_pred, y_real)[0]
    rmse = np.sqrt(mean_squared_error(y_pred, y_real))
    toc = time.perf_counter()
    print('Final RMSE: %.3f, Rp: %.4f\nElapsed time: %.1f [min]'%(rmse, pcc, (toc-tic)/60))

def skempi_pretrain(args):
    tic = time.perf_counter()

    # setup device cuda or cpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # setup dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # num_workers=0?

    num_channel = 9

    X_val = np.load('./skempi2/X_val_S8338.npy')
    normalizer1 = joblib.load('model/normalizer_mega.pkl')
    X_val = normalizer1.transform(X_val)

    X_val_Lap_ESM = np.load('./skempi2/X_val_S8338_Lap_ESM.npy')
    normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
    X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

    X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)

    y_val = np.load('./skempi2/Y_S8338.npy').reshape((-1, 1))
    print('The data shape', X_val.shape, ', label size', y_val.shape)

    X_val, y_val = randomize_features(X_val, y_val)
    X_dms, y_dms = load_data(args)

    y_pred = np.array([])
    y_real = np.array([])
    
    # setup dataloader
    train_dataset = TopLapNet(X_val, y_val)
    valid_loader = []
    valid_loader.append(DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs))
    for i in range(1, num_channel):
        X_tmp = TopLapNet(X_dms[i-1], y_dms[i-1])
        valid_loader.append(DataLoader(dataset=X_tmp, batch_size=args.batch_size, **kwargs))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs)
        
    input_layer = X_val.shape[1]
    num_channel = 9
    #num_channel = 8
    hiden_layer = [int(i) for i in args.layers.split(',')]
    model = MultitaskModule(input_layer, hiden_layer, num_channel)
    #print(model.hiden_layers)
    freeze_args = [bool(int(i)) for i in args.freeze.split(',')]
    model.input_layer.requires_grad_(freeze_args[0])
    for i in range(1, len(freeze_args)):
        model.hiden_layers[i-1].requires_grad_(freeze_args[i])

    print('loading multitask_C%d model<<<<<<<<<<<<<<<<<<'%num_channel)
    model.load_state_dict(torch.load(f'model/multitask_C{num_channel}_pretrained_DMS.pkl'))
    model = model.to(device)

    criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), 
                        lr=args.lr, 
                        momentum=args.momentum, 
                        weight_decay=args.weight_decay)
    lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    for epoch in range(args.epochs):
        #for task_idx in range(num_channel):
        train(model, device, train_loader, criterion, optimizer, 0)
        for task_idx in range(num_channel):
            if epoch%args.log_interval == 0:
                if task_idx==0: print('epoch %d >>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
                test(model, device, valid_loader[task_idx], str(epoch), task_idx)
                if task_idx==num_channel-1 and epoch!=0:
                    print(f'epoch {epoch}, model saved multitask_C{num_channel}_pretrained')
                    torch.save(model.state_dict(), f'model/multitask_C{num_channel}_pretrained.pkl')
        lr_adjust.step()

    for task_idx in range(num_channel):
        test(model, device, valid_loader[task_idx], str(epoch), task_idx)
        if task_idx==num_channel-1:
            print(f'epoch {epoch}, model saved multitask_C{num_channel}_pretrained')
            torch.save(model.state_dict(), f'model/multitask_C{num_channel}_pretrained.pkl')

    toc = time.perf_counter()
    print('Elapsed time: %.1f [min]'%((toc-tic)/60))

def prediction(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('The prediction evaluation for '+args.pred+':')
    if args.pred == 'CAPRI':
        X_val_3R2X = np.load('./3R2X/X_val_3R2X.npy')
        X_val_4EEF = np.load('./4EEF/X_val_4EEF.npy')
        X_val = np.concatenate((X_val_3R2X, X_val_4EEF), axis=0)

        normalizer1 = joblib.load('model/normalizer_mega.pkl')
        X_val = normalizer1.transform(X_val)

        X_val_3R2X_Lap_ESM = np.load('./3R2X/X_val_3R2X_Lap_ESM.npy')
        X_val_4EEF_Lap_ESM = np.load('./4EEF/X_val_4EEF_Lap_ESM.npy')
        normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
        X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

        X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)

        y_val_3R2X = np.load('./3R2X/Y_3R2X.npy').reshape((-1, 1))
        y_val_4EEF = np.load('./4EEF/Y_4EEF.npy').reshape((-1, 1))
        yreal = np.concatenate((y_val_3R2X, y_val_4EEF), axis=0)
    elif args.pred == 'skempi2':
        X_val = np.load('./skempi2/X_val_S8338.npy')
        normalizer1 = joblib.load('model/normalizer_mega.pkl')
        X_val = normalizer1.transform(X_val)

        X_val_Lap_ESM = np.load('./skempi2/X_val_S8338_Lap_ESM.npy')
        normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
        X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

        X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)

        yreal = np.load('./skempi2/Y_S8338.npy').reshape((-1, 1))
    else:
        X_val = np.load(f'./{args.pred}/X_val_{args.pred}.npy')
        normalizer1 = joblib.load('model/normalizer_mega.pkl')
        X_val = normalizer1.transform(X_val)

        X_val_Lap_ESM = np.load(f'./{args.pred}/X_val_{args.pred}_Lap_ESM.npy')
        normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
        X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

        X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)

        yreal = np.load(f'./{args.pred}/Y_{args.pred}.npy').reshape((-1, 1))

    print(f'X_val.shape = {X_val.shape}, y_val.shape = {yreal.shape}')
    hiden_layer = [int(i) for i in args.layers.split(',')]
    input_layer = X_val.shape[1]
    num_channel = 9
    #num_channel = 8
    model = MultitaskModule(input_layer, hiden_layer, num_channel)
    model.load_state_dict(torch.load(f'model/multitask_C{num_channel}_pretrained.pkl'))
    model.to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    model.eval()
    with torch.no_grad():
        ypred = model(X_val)[:, 0].view(-1, 1).cpu().numpy().ravel()
        yreal = yreal.ravel()
    #if not args.single:
    #index = np.linspace(0, yreal.shape[0]-2, int(yreal.shape[0]/2), dtype=np.int16)
    #ypred = ypred[index]
    #yreal = yreal[index]
    print(f'y_val.shape = {yreal.shape}')

    fp = open(f'./{args.pred}/{args.pred}_pred.txt', 'w')
    for i in range(len(yreal)):
        fp.write(f'{ypred[i]} {yreal[i]}\n')
    fp.close()
    yavg = yreal.mean()
    ymin = yreal.min()
    ymax = yreal.max()
    ypred = (ypred+yavg)/(ymax-ymin)
    global_free, total_occupied = torch.cuda.mem_get_info()
    print('Check memory usage', global_free//1024**3, total_occupied//1024**3)
    RMSD = np.sqrt(mean_squared_error(yreal, ypred))
    print('RMSD: %f'%(RMSD))
    Rp   = scipy.stats.pearsonr(yreal, ypred)
    print('Rp  : %f'%(Rp[0]))
    Ktau = scipy.stats.kendalltau(yreal, ypred)
    print('Ktau: %f'%(Ktau[0]))
    return

def finetune(args):
    tic = time.perf_counter()

    # setup device cuda or cpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # setup dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # num_workers=0?

    num_channel = 9

    X_val = np.load(f'./{args.finetune}/X_val_{args.finetune}.npy')
    normalizer1 = joblib.load('model/normalizer_mega.pkl')
    X_val = normalizer1.transform(X_val)

    X_val_Lap_ESM = np.load(f'./{args.finetune}/X_val_{args.finetune}_Lap_ESM.npy')
    normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
    X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

    X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)
    
    y_val = np.load(f'./{args.finetune}/Y_{args.finetune}.npy').reshape((-1, 1))
    print('The data shape', X_val.shape, ', label size', y_val.shape)

    X_skempi = np.load('./skempi2/X_val_S8338.npy')
    normalizer1 = joblib.load('model/normalizer_mega.pkl')
    X_skempi = normalizer1.transform(X_skempi)

    X_skempi_Lap_ESM = np.load('./skempi2/X_val_S8338_Lap_ESM.npy')
    normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
    X_skempi_Lap_ESM = normalizer2.transform(X_skempi_Lap_ESM)

    X_skempi = np.concatenate((X_skempi, X_skempi_Lap_ESM), axis=1)

    y_skempi = np.load('./skempi2/Y_S8338.npy').reshape((-1, 1))
    #print('The data shape', X_skempi.shape, ', label size', y_skempi.shape)

    X_val, y_val = randomize_features(X_val, y_val)
    X_skempi, y_skempi = randomize_features(X_skempi, y_skempi)
    X_dms, y_dms = load_data(args)

    y_pred = np.array([])
    y_real = np.array([])
    
    # setup dataloader
    train_dataset = TopLapNet(X_val, y_val)
    valid_dataset = TopLapNet(X_skempi, y_skempi)
    valid_loader = []
    valid_loader.append(DataLoader(dataset=valid_dataset, batch_size=args.batch_size, **kwargs))
    for i in range(1, num_channel):
        X_tmp = TopLapNet(X_dms[i-1], y_dms[i-1])
        valid_loader.append(DataLoader(dataset=X_tmp, batch_size=args.batch_size, **kwargs))
    valid_loader.append(DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs)
        
    input_layer = X_val.shape[1]
    num_channel = 9
    #num_channel = 8
    hiden_layer = [int(i) for i in args.layers.split(',')]
    model = MultitaskModule(input_layer, hiden_layer, num_channel)
    #print(model.hiden_layers)
    freeze_args = [bool(int(i)) for i in args.freeze.split(',')]
    model.input_layer.requires_grad_(freeze_args[0])
    for i in range(1, len(freeze_args)):
        model.hiden_layers[i-1].requires_grad_(freeze_args[i])

    print('loading multitask_C%d model<<<<<<<<<<<<<<<<<<'%num_channel)
    model.load_state_dict(torch.load(f'model/multitask_C{num_channel}_pretrained.pkl'))
    model.add_channel(hiden_layer, num_channel)
    model = model.to(device)

    #print(model)

    criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), 
                        lr=args.lr, 
                        momentum=args.momentum, 
                        weight_decay=args.weight_decay)
    lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    for epoch in range(args.epochs):
        #for task_idx in range(num_channel):
        train(model, device, train_loader, criterion, optimizer, num_channel)
        for task_idx in range(num_channel+1):
            if epoch%args.log_interval == 0:
                if task_idx==0: print('epoch %d >>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
                test(model, device, valid_loader[task_idx], str(epoch), task_idx)
                #test(model, device, train_loader, str(epoch), task_idx)
                if task_idx==num_channel and epoch!=0:
                    print(f'epoch {epoch}, model saved multitask_C{num_channel}_finetuned_{args.finetune}')
                    torch.save(model.state_dict(), f'model/multitask_C{num_channel}_finetuned_{args.finetune}.pkl')
        lr_adjust.step()

    print(f'epoch {epoch}, model saved multitask_C{num_channel}_finetuned_{args.finetune}')
    torch.save(model.state_dict(), f'model/multitask_C{num_channel}_finetuned_{args.finetune}.pkl')

    X_val = np.load(f'./{args.finetune}/X_val_{args.finetune}.npy')
    normalizer1 = joblib.load('model/normalizer_mega.pkl')
    X_val = normalizer1.transform(X_val)

    X_val_Lap_ESM = np.load(f'./{args.finetune}/X_val_{args.finetune}_Lap_ESM.npy')
    normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
    X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

    X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)
    
    y_val = np.load(f'./{args.finetune}/Y_{args.finetune}.npy').reshape((-1, 1))
    print('The data shape', X_val.shape, ', label size', y_val.shape)

    X_val = torch.from_numpy(X_val).float().to(device)
    model.eval()
    with torch.no_grad():
        ypred = model(X_val)[:, 0].view(-1, 1).cpu().numpy().ravel()
        yreal = y_val.ravel()
    #if not args.single:
    #index = np.linspace(0, yreal.shape[0]-2, int(yreal.shape[0]/2), dtype=np.int16)
    #ypred = ypred[index]
    #yreal = yreal[index]
    print(f'y_val.shape = {yreal.shape}')

    fp = open(f'./{args.finetune}/{args.finetune}_pred_finetune.txt', 'w')
    for i in range(len(yreal)):
        fp.write(f'{ypred[i]} {yreal[i]}\n')
    fp.close()
    yavg = yreal.mean()
    ymin = yreal.min()
    ymax = yreal.max()
    ypred = (ypred+yavg)/(ymax-ymin)
    global_free, total_occupied = torch.cuda.mem_get_info()
    print('Check memory usage', global_free//1024**3, total_occupied//1024**3)
    RMSD = np.sqrt(mean_squared_error(yreal, ypred))
    print('RMSD: %f'%(RMSD))
    Rp   = scipy.stats.pearsonr(yreal, ypred)
    print('Rp  : %f'%(Rp[0]))
    Ktau = scipy.stats.kendalltau(yreal, ypred)
    print('Ktau: %f'%(Ktau[0]))

    toc = time.perf_counter()
    print('Elapsed time: %.1f [min]'%((toc-tic)/60))
    return

def predict_DMS(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('The prediction evaluation for '+args.DMS_data+':')
    data = args.DMS_data.split("_")[0]+"_"+args.DMS_data.split("_")[1]
    #mutid = args.pred_mutid.split("_")[1]

    os.chdir(f'./{data}/')

    #seq_list = dataset_list(f'./{data}/{data[:4]}_RBD.txt')
    #print(seq_list)

    rows = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
    index = []
    #start = int(seq_list[0][-3])
    #end = int(seq_list[-1][-3])

    structure = PDBParser().get_structure(f'{data}', f'./{data[:4]}.pdb')    

    model = structure[0]
    chain = model[args.DMS_data.split("_")[2]]

    if not os.path.exists(f'./DMS_data/X_val_{data}.npy') or not os.path.exists(f'./DMS_data/X_val_{data}_Lap_ESM.npy'):
        feat_aux = []
        feat_ph0 = []
        feat_ph12 = []
        feat_fri = []
        feat_lap = []
        feat_esm = []
        for i in chain.get_residues():
            if i.get_resname() in d.keys():
                res_code = d[i.get_resname()]
                res_id = i.get_full_id()[3][1]
                for j in rows:
                    if j != res_code:
                        #print(f'{res_code}_{res_id}_{j}')
                        os.chdir("features/{}_{}_{}_{}_{}".format(data[:4], args.DMS_data.split("_")[2], res_code, str(res_id), j))
                        
                        PDBid, Chain, resWT, resID, resMT = data[:4], args.DMS_data.split("_")[2], res_code, str(res_id), j  
                        index.append([PDBid, Chain, resWT, resID, resMT])
                        filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
                        #filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT

                        aux = np.load(filename+'_aux.npy', allow_pickle=True)
                        #aux_inv = np.load(filename_inv+'_aux.npy', allow_pickle=True)

                        fri = np.load(filename+'_FRI.npy', allow_pickle=True)
                        #fri_inv = np.load(filename_inv+'_FRI.npy', allow_pickle=True)

                        ph0 = np.load(filename+'_PH0.npy', allow_pickle=True)
                        #ph0_inv = np.load(filename_inv+'_PH0.npy', allow_pickle=True)

                        ph12 = np.load(filename+'_PH12.npy', allow_pickle=True)
                        #ph12_inv = np.load(filename_inv+'_PH12.npy', allow_pickle=True)

                        lap = np.load(filename+'_Lap_b.npy', allow_pickle=True)
                        #lap_inv = np.load(filename_inv+'_Lap_b.npy', allow_pickle=True)

                        esm = np.load(filename+'_seq.npy', allow_pickle=True)
                        #esm_inv = np.load(filename_inv+'_seq.npy', allow_pickle=True)
                        
                        os.chdir("../../")

                        feat_aux.append(aux)
                        #feat_aux.append(aux_inv)
                        feat_ph0.append(ph0)
                        #feat_ph0.append(ph0_inv)
                        feat_fri.append(fri)
                        #feat_fri.append(fri_inv)
                        feat_ph12.append(ph12)
                        #feat_ph12.append(ph12_inv)
                        feat_esm.append(esm)
                        #feat_esm.append(esm_inv)
                        feat_lap.append(lap)
                        #feat_lap.append(lap_inv)

        feat_aux = np.array(feat_aux)
        feat_fri = np.array(feat_fri)
        feat_ph0 = np.array(feat_ph0)
        feat_ph12 = np.array(feat_ph12)
        feat_lap = np.array(feat_lap)
        feat_esm = np.array(feat_esm)

        print(np.shape(feat_aux))
        print(np.shape(feat_fri))
        print(np.shape(feat_ph0))
        print(np.shape(feat_ph12))
        print(np.shape(feat_lap))
        print(np.shape(feat_esm))

        os.chdir('../')

        if not os.path.exists('./DMS_data/'):
            os.mkdir('./DMS_data/')

        np.save(f'./DMS_data/X_{data}_aux.npy', feat_aux)
        np.save(f'./DMS_data/X_{data}_FRI.npy', feat_fri)
        np.save(f'./DMS_data/X_{data}_PH0.npy', feat_ph0)
        np.save(f'./DMS_data/X_{data}_PH12.npy', feat_ph12)
        np.save(f'./DMS_data/X_{data}_Lap_b.npy', feat_lap)
        np.save(f'./DMS_data/X_{data}_ESM.npy', feat_esm)

        X_val1 = np.load(f'./DMS_data/X_{data}_aux.npy')
        X_val2 = np.load(f'./DMS_data/X_{data}_FRI.npy')
        X_val3 = np.load(f'./DMS_data/X_{data}_PH0.npy')
        X_val4 = np.load(f'./DMS_data/X_{data}_PH12.npy')
        X_val5 = np.load(f'./DMS_data/X_{data}_ESM.npy')
        X_val6 = np.load(f'./DMS_data/X_{data}_Lap_b.npy')
        X_val = np.concatenate((X_val1, X_val2), axis=1)
        X_val = np.concatenate((X_val,  X_val3), axis=1)
        X_val = np.concatenate((X_val,  X_val4), axis=1)
        np.save(f'./DMS_data/X_val_{data}.npy', X_val)

        X_val = np.concatenate((X_val5,  X_val6), axis=1)
        np.save(f'./DMS_data/X_val_{data}_Lap_ESM.npy', X_val)
    
    X_val = np.load(f'./DMS_data/X_val_{data}.npy')
    normalizer1 = joblib.load('model/normalizer_mega.pkl')
    X_val = normalizer1.transform(X_val)

    X_val_Lap_ESM = np.load(f'./DMS_data/X_val_{data}_Lap_ESM.npy')
    normalizer2 = joblib.load('model/normalizer_Lap_ESM.pkl')
    X_val_Lap_ESM = normalizer2.transform(X_val_Lap_ESM)

    X_val = np.concatenate((X_val, X_val_Lap_ESM), axis=1)

    print(f'X_val.shape = {X_val.shape}')
    hiden_layer = [int(i) for i in args.layers.split(',')]
    input_layer = X_val.shape[1]
    num_channel = 9
    #num_channel = 8
    model = MultitaskModule(input_layer, hiden_layer, num_channel+1)
    model.load_state_dict(torch.load(f'model/multitask_C{num_channel}_finetuned_{data}.pkl'))
    model.to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    model.eval()
    with torch.no_grad():
        ypred = model(X_val)[:, 0].view(-1, 1).cpu().numpy().ravel()
        yreal = model(X_val)[:, -1].view(-1, 1).cpu().numpy().ravel()

    print(f'ypred.shape = {ypred.shape}, index.shape = {np.array(index).shape}')
    fp = open(f'./DMS_data/{data}_pred.txt', 'w')
    for i in range(len(ypred)):
        PDBid, Chain, resWT, resID, resMT  = index[i][0], index[i][1], index[i][2], index[i][3], index[i][4]
        fp.write(f'{PDBid},{Chain},{resWT},{resID},{resMT},{ypred[i]},{yreal[i]}\n')
    fp.close()
    return


if __name__=='__main__':
    from datetime import datetime, date
    today = date.today()
    print("Today's date:", today)
    now = datetime.now()
    current_time = now.strftime('%H:%M:%S')
    print('Current Time =', current_time)
    parser = argparse.ArgumentParser(description='multitask of log2 and BFE')
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0, help='SGD weight decay (default: 0)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--layers', type=str, default='15000,15000,15000,15000,15000,15000', help='neural network layers and neural numbers')
    parser.add_argument('--continue_train', type=bool, default=False, help='run training')
    parser.add_argument('--prediction', type=bool, default=False, help='prediction')
    parser.add_argument('--pred', type=str, default='7XB0_RBD')
    parser.add_argument('--cv', type=bool, default=False, help='cv')
    parser.add_argument('--cv_type', type=str, default='skempi2', help='skempi2 or alphafold')
    parser.add_argument('--model', type=str, default='C9', help='prediction model')
    #parser.add_argument('--leave_who', type=str, default='skempi2', help='prediction')
    parser.add_argument('--normalizer1_name', type=str, default='_mega_alphafold', help='mega dataset')
    parser.add_argument('--normalizer2_name', type=str, default='_Lap_ESM_alphafold', help='Lap and ESM dataset')
    parser.add_argument('--freeze', type=str, default='0,0,0,1,1', help='freeze weights and biases of hidden layers')
    parser.add_argument('--skempi_pretrain', type=bool, default=False, help='finetune DMS with SKEMPI2')
    parser.add_argument('--ft', type=bool, default=False, help='finetune pretrained model with DMS')
    parser.add_argument('--finetune', type=str, default='7ZF7_RBD', help='finetune model with DMS data')
    parser.add_argument('--debug', type=bool, default=False, help='debugging channel')
    parser.add_argument('--DMS_data', type=str, default='8HG0_dACE2_B', help='predict full DMS data')
    parser.add_argument('--pred_DMS', type=bool, default=False, help='predict full DMS data')

    args = parser.parse_args()
    print(args)

    if args.cv == True:
        if args.cv_type == 'skempi2':
            skempi_cv(args)
        else:
            alphafold_cv(args)
    elif args.skempi_pretrain == True:
        skempi_pretrain(args)
    elif args.prediction == True:
        prediction(args)
    elif args.ft == True:
        finetune(args)
    elif args.debug == True:
        input_layer = 10541
        num_channel = 9
        #num_channel = 8
        hiden_layer = [int(i) for i in args.layers.split(',')]
        model = MultitaskModule(input_layer, hiden_layer, num_channel)
        #print(model.hiden_layers)
        freeze_args = [bool(int(i)) for i in args.freeze.split(',')]
        model.input_layer.requires_grad_(freeze_args[0])
        for i in range(1, len(freeze_args)):
            model.hiden_layers[i-1].requires_grad_(freeze_args[i])
        model.add_channel(hiden_layer, num_channel)
        print(model)
        print(model.input_layer.weight)
        print(model.input_layer.bias)
        for i in range(5):
            print(model.hiden_layers[i].weight)
            print(model.hiden_layers[i].bias)
        print(model.output_layer.weight)
        print(model.output_layer.bias)
    elif args.pred_DMS == True:
        predict_DMS(args)
    else:
        generate_model(args)

    now = datetime.now()
    current_time = now.strftime('%H:%M:%S')
    print('Finish Time =', current_time)
