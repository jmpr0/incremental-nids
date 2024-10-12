import sys
sys.path.append('../')
import json
from box import Box
import utils
from datasets.data_loader import get_loaders
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import math
from collections import OrderedDict
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score, recall_score, roc_auc_score
import seaborn as sn
from glob import glob
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from matplotlib.colors import LogNorm, Normalize
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import pickle

file_dict = {
    'acc': 'per_class_metrics_metrics_balanced_accuracy_score_macro-material.parquet',
    'f1' : 'per_class_metrics_metrics_f1_score_macro-material.parquet',
    'accdrop': 'per_class_metrics_metrics_balanced_accuracy_score_macro intransigence-material.parquet',
    'f1drop': 'per_class_metrics_metrics_f1_score_macro intransigence-material.parquet',
    'prec': 'per_class_metrics_metrics_precision_score_macro-material.parquet',
    'rec': 'per_class_metrics_metrics_recall_score_macro-material.parquet'
}


def get_appr_params(base_path, ls_factor=0, lambd=-1, T=2):
    dict_appr_params  = {} 
    alpha, beta = None, None
    results_files = [elem for elem in glob(f'{base_path}/results/*') if (('outputs_targets_features_' in elem) \
                        and elem.split('/')[-1].split('-')[0].split('_')[-1]=='1')]
    if len(results_files)==0:
        results_files = [elem for elem in glob(f'{base_path}/results/*') if 'per_class_metrics' in elem]    
    for out_path in results_files:    
        timestamp = out_path.split('.')[0].split('-')[-1].split('_')[0]
        args_file = f'{base_path}/args-{timestamp}.txt'
        stdout_file = f'{base_path}/stdout-{timestamp}.txt'
        with open(stdout_file) as f:
            lines = f.readlines()
        with open(args_file) as f:
            args = Box(json.loads(f.read()))
        ls = float(args.get('ls_factor', default = 0))
        lamb = float(args.get('lamb', default = -1))
        t = float(args.get('T', default = 2))
        if ls==ls_factor and lambd==lamb and t==T:
            class_order = [int(elem) for elem in lines[1].split(':')[-1].replace('[','').replace(']','').replace('','').split(',')]
            for l in lines:
                if 'BiC training for Task 1' in l:
                    alpha=float(l.split('alpha=')[-1].split(',')[0])
                    beta=float(l.split('beta=')[-1])
            if args.last_class!=-1:
                dict_appr_params[args.last_class]={'app':classes[args.last_class], 'alpha':alpha, 'beta':beta, 
                                               'timestamp':timestamp, 'class_order':class_order}
            else:
                dict_appr_params[args.seed]={'alpha':alpha, 'beta':beta, 
                                               'timestamp':timestamp, 'class_order':class_order}
    return dict_appr_params       

def get_appr_params_memory(base_path, ls_factor=0, lambd=-1, T=2):
    dict_appr_params  = {} 
    alpha, beta = None, None
    results_files = [elem for elem in glob(f'{base_path}/results/*') if (('outputs_targets_features_' in elem) \
                        and elem.split('/')[-1].split('-')[0].split('_')[-1]=='0')]
    if len(results_files)==0:
        results_files = [elem for elem in glob(f'{base_path}/results/*') if 'per_class_metrics' in elem]    
    for out_path in results_files:    
        timestamp = out_path.split('.')[0].split('-')[-1].split('_')[0]
        args_file = f'{base_path}/args-{timestamp}.txt'
        stdout_file = f'{base_path}/stdout-{timestamp}.txt'
        with open(stdout_file) as f:
            lines = f.readlines()
        with open(args_file) as f:
            args = Box(json.loads(f.read()))
        ls = float(args.get('ls_factor', default = 0))
        lamb = float(args.get('lamb', default = -1))
        t = float(args.get('T', default = 2))
        if ls==ls_factor and lambd==lamb and t==T:
            class_order = [int(elem) for elem in lines[1].split(':')[-1].replace('[','').replace(']','').replace('','').split(',')]
            for l in lines:
                if 'BiC training for Task 1' in l:
                    alpha=float(l.split('alpha=')[-1].split(',')[0])
                    beta=float(l.split('beta=')[-1])
            if args.last_class!=-1:
                dict_appr_params[args.last_class]={'app':classes[args.last_class], 'alpha':alpha, 'beta':beta, 
                                               'timestamp':timestamp, 'class_order':class_order}
            else:
                dict_appr_params[args.seed]={'alpha':alpha, 'beta':beta, 
                                               'timestamp':timestamp, 'class_order':class_order}
    return dict_appr_params       

def get_appr_params_fscil(base_path, seed, s, ls_factor=0, lambd=1, T=2):
    dict_appr_params  = {} 
    alpha, beta = None, None
    results_files = [elem for elem in glob(f'{base_path}/results/*') if (('outputs_targets_features_' in elem) \
                        and elem.split('/')[-1].split('-')[0].split('_')[-1]=='1')]
    #print(len(results_files))
    #input()
    if len(results_files)==0:
        results_files = [elem for elem in glob(f'{base_path}/results/*') if 'per_class_metrics' in elem]    
    for out_path in results_files:    
        timestamp = out_path.split('.')[0].split('-')[-1].split('_')[0]
        args_file = f'{base_path}/args-{timestamp}.txt'
        stdout_file = f'{base_path}/stdout-{timestamp}.txt'
        with open(stdout_file) as f:
            lines = f.readlines()
        with open(args_file) as f:
            args = Box(json.loads(f.read()))
        ls = float(args.get('ls_factor', default = 0))
        lamb = float(args.get('lamb', default = -1))
        t = float(args.get('T', default = 2))
        if ls==ls_factor and lambd==lamb and t==T:
            #class_order = [int(elem) for elem in lines[1].split(':')[-1].replace('[','').replace(']','').replace('','').split(',')]
            class_order = None
            for l in lines:
                if 'BiC training for Task 1' in l:
                    alpha=float(l.split('alpha=')[-1].split(',')[0])
                    beta=float(l.split('beta=')[-1])
            if args.last_class!=-1:
                dict_appr_params[args.last_class]={'app':classes[args.last_class], 'alpha':alpha, 'beta':beta, 
                                               'timestamp':timestamp, 'class_order':class_order}
            else:
                if seed == args.seed and s == args.shots and args.fseed!=100:
                    dict_appr_params[args.fseed]={'alpha':alpha, 'beta':beta, 
                                               'timestamp':timestamp, 'class_order':class_order, 'shots':s}
    return dict_appr_params         

classes = ['8 Ball Pool','AccuWeather','Amazon','Booking','Comics','Diretta','Dropbox','Duolingo','Facebook',
           'Flipboard','FourSquare','Google Drive','Groupon','Hangouts','Instagram','Linkedin','Messenger',
           'Musixmatch','OneDrive','OneFootball','Pinterest','Playstore','Reddit','Skype','Slither.io',
           'SoundCloud','Spotify','Subito','Telegram','Trello','TripAdvisor','Tumblr','Twitter','Uber','Viber'
           ,'Waze','Wish','Youtube','eBay', 'ilMeteo']


def plot_cm(targets, pred, appr=None, title=None, last_class=None, binary=False, order=False, 
            cv=False, pred_scratch=None, vmin=None, vmax=None):
    
    annotation = False
    norm = LogNorm(0.01)
    norm = None
    normalize = 'true'
    if binary:
        norm, annotation, normalize = None, True, None
        first_ep = [i for i,x in enumerate(targets) if x<20] if cv else np.where(targets!=last_class)[0]
        correct_classes_old = len(np.where(np.array(targets)[first_ep]==np.array(pred)[first_ep])[0])     
        if cv:
            second_ep = [i for i,x in enumerate(targets) if x>=20]
            correct_classes_new = len(np.where(np.array(targets)[second_ep]==np.array(pred)[second_ep])[0])
            
        targets_bin = np.array([0 if x<20 else 1 for x in targets]) if cv \
                                else np.array([0 if x!=last_class else 1 for x in targets])
        
        pred_bin = np.array([0 if x<20 else 1 for x in pred]) if cv \
                                else np.array([0 if x!=last_class else 1 for x in pred])
        
        c_old_ind = np.where((targets_bin==pred_bin) & (pred_bin==0))[0]
        correct_classes_scratch = len(np.where(np.array(targets)[c_old_ind]==np.array(pred_scratch)[c_old_ind])[0])
        targets=targets_bin
        pred=pred_bin
                       
    cm = confusion_matrix(targets, pred, normalize=normalize)
    ax = sn.heatmap(cm*100, annot=annotation, yticklabels=False, xticklabels=False, linewidths=0.8, linecolor='whitesmoke', square=True,
                    vmin=0, vmax=100, norm = norm, cmap=sn.color_palette("YlOrBr", as_cmap=True))
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('darkorange')
    
    if binary:
        plt.text(0.25, 0.65, f'Classes: {correct_classes_old}', horizontalalignment='left', size='medium', color='black')
        plt.text(0.25, 0.75, f'Scratch: {correct_classes_scratch}', horizontalalignment='left', size='medium', color='black')
        if cv:
            plt.text(1.3, 1.65, f'Classes: {correct_classes_new}', horizontalalignment='left', size='medium', color='black')

    #ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    Path(f'results/Confusion_Matrices/{appr}').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'results/Confusion_Matrices/{appr}/{title}.pdf')
    print('saved at', f'./results/Confusion_Matrices/{appr}/{title}.pdf')
    plt.show() 
    #plt.clf()
    
def relu(x):
    return np.maximum(0, x)

#bic_model_nobc_originale
device = torch.device('cpu')
#utils.seed_everything(seed=args.seed)

class BIC(nn.Module):       
    def __init__(self, nc_base):
        super(BIC, self).__init__()
        self.alpha = 1
        self.beta = 0
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4,2), stride=(1,1), padding=0)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d((3,2), (1,1), 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,2), stride=(1, 1), padding=0)        
        self.max2 = nn.MaxPool2d((3,1), stride=(1,1), padding=0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(2560,200)        
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(200, nc_base)
        self.fc3 = nn.Linear(200, 7-nc_base)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = F.pad(x, (0, 1) + (1, 2))
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.pad(x, (0, 1) + (1, 1))
        x = self.max1(x)
        x = self.bn1(x)
        x = F.pad(x, (0,1) + (1,2))
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.pad(x, (0,0) + (1,1))
        x = self.max2(x)
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x1 = self.fc2(x)
        x2 = self.fc3(x)
        x2= self.alpha*x2+self.beta
        x = torch.cat((x1,x2), axis=1)
        x4 = x[None, :]
        return x4
    
    def feature_extr(self, x):
        x = F.pad(x, (0, 1) + (1, 2))
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.pad(x, (0, 1) + (1, 1))
        x = self.max1(x)
        x = self.bn1(x)
        x = F.pad(x, (0,1) + (1,2))
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.pad(x, (0,0) + (1,1))
        x = self.max2(x)
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        return x
    
# model_bic_nobc = BIC(39).to(device)
# model_bic_nobc_20 = BIC(20).to(device)
device = torch.device('cpu')
#utils.seed_everything(seed=args.seed)

#scratch model 39
device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self, nc_base):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4,2), stride=(1,1), padding=0)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d((3,2), (1,1), 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,2), stride=(1, 1), padding=0)        
        self.max2 = nn.MaxPool2d((3,1), stride=(1,1), padding=0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(2560,200)        
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(200, nc_base)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.pad(x, (0, 1) + (1, 2))
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.pad(x, (0, 1) + (1, 1))
        x = self.max1(x)
        x = self.bn1(x)
        x = F.pad(x, (0,1) + (1,2))
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.pad(x, (0,0) + (1,1))
        x = self.max2(x)
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        return x
    
    def feature_extr(self, x):
        x = F.pad(x, (0, 1) + (1, 2))
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.pad(x, (0, 1) + (1, 1))
        x = self.max1(x)
        x = self.bn1(x)
        x = F.pad(x, (0,1) + (1,2))
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.pad(x, (0,0) + (1,1))
        x = self.max2(x)
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        #x = self.fc2(x)
        #x = self.softmax(x)
        return x
    
scratch_model_39 = Net(39).to(device)
scratch_model_20 = Net(20).to(device)
scratch_model_40 = Net(40).to(device)

# TODO: Modificare i valori nel modello per il caso 3 feature in input
class BIC_3f(nn.Module):       
    def __init__(self, nc_all, nc_base):
        super(BIC_3f, self).__init__()
        self.alpha = 1
        self.beta = 0
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4,2), stride=(1,1), padding=0)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d((3,2), (1,1), 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,2), stride=(1, 1), padding=0)        
        self.max2 = nn.MaxPool2d((3,1), stride=(1,1), padding=0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1920,200)        
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(200, nc_base)
        self.fc3 = nn.Linear(200, nc_all-nc_base)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = F.pad(x, (0, 1) + (1, 2))
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.pad(x, (0, 1) + (1, 1))
        x = self.max1(x)
        x = self.bn1(x)
        x = F.pad(x, (0,1) + (1,2))
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.pad(x, (0,0) + (1,1))
        x = self.max2(x)
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x1 = self.fc2(x)
        x2 = self.fc3(x)
        x2= self.alpha*x2+self.beta
        x = torch.cat((x1,x2), axis=1)
        x4 = x[None, :]
        return x4
    
    def feature_extr(self, x):
        x = F.pad(x, (0, 1) + (1, 2))
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.pad(x, (0, 1) + (1, 1))
        x = self.max1(x)
        x = self.bn1(x)
        x = F.pad(x, (0,1) + (1,2))
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.pad(x, (0,0) + (1,1))
        x = self.max2(x)
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        return x