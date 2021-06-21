import pandas as pd
import numpy as np
import torch
from torchvision import models

import cv2 
import pydicom
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms

from torch import nn
import warnings
from efficientnet_pytorch import EfficientNet

import random
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage

def get_tab(df):
    vector = [(df.Weeks.values[0] - 30 )/30]
    
    if df.Sex.values[0] == 'Male':
      vector.append(0)
    else:
      vector.append(1)
    
    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0])
    return np.array(vector) 

def collate_fn_test(b):
    imgs,tabs = zip(*b)
    return (torch.stack(imgs).float(),torch.stack(tabs).float())           

def get_img(record):
    d = pydicom.dcmread(record)
    return cv2.resize(d.pixel_array / 2**11, (512, 512))        

class OSIC_Model(nn.Module):
    def __init__(self,eff_name='efficientnet-b0'):
        super().__init__()
        self.input = nn.Conv2d(1,3,kernel_size=3,padding=1,stride=2)
        self.bn = nn.BatchNorm2d(3)
        self.model = EfficientNet.from_pretrained(eff_name)
        self.model._fc = nn.Linear(1536, 500, bias=True)
        self.meta = nn.Sequential(nn.Linear(4, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500,250),
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.output = nn.Linear(500+250, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x,tab):
        x = self.relu(self.bn(self.input(x)))
        x = self.model(x)
        tab = self.meta(tab)
        x = torch.cat([x, tab],dim=1)
        return self.output(x)

class Dataset:
    def __init__(self, df, tabular, targets, Patient , filename , mode):
        self.df = df
        self.tabular = tabular
        self.targets = targets
        self.mode = mode
        self.filename=filename
        self.Patient=Patient
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row = self.df.loc[idx,:]
        pid = row['Patient']
        record='/Users/saikishorehr/Documents/FYP/PulmonaryFibrosis/media/'+ self.Patient +'/'+self.filename
        try:                       
            img = get_img(record)
            img = self.transform(img)
            tab = torch.from_numpy(self.tabular[pid]).float()
            if self.mode == 'train':
                target = torch.tensor(self.targets[pid])
                return (img,tab),target
            else:
                return (img,tab)
        except Exception as e:
            print(e)

def index(request):
    return render(request,'index.html')

@login_required  
def pulmonaryfibrosis(request):
    dev=torch.device('cpu')
    value = ''
    warnings.simplefilter('ignore')

    if request.method == 'POST':
        Patient1 = request.POST['patient']
        FWeeks1 = float(request.POST['fweeks'])
        FVC1 = float(request.POST['fvc'])
        Percent1 = float(request.POST['percent'])
        Age1 = float(request.POST['age'])
        Sex1 = request.POST['sex']
        SmokingStatus1 = request.POST['smokingstatus']
        myfile=request.FILES['image_ct']

        PWeeks1=int(request.POST['pweeks']) + 12
        
        fs = FileSystemStorage(location='/Users/saikishorehr/Documents/FYP/PulmonaryFibrosis/media/'+ Patient1 +'/')  
        filename = fs.save(myfile.name, myfile)

        data={
           "Patient":[Patient1],
           "Weeks":[FWeeks1],
           "FVC":[FVC1],
           "Percent":[Percent1],
           "Age":[Age1],
           "Sex":[Sex1],
           "SmokingStatus":[SmokingStatus1]
        }
        test_df=pd.DataFrame(data)
        test_data= [] 
        models = {}
        for i in range(2):
            models[i] = OSIC_Model('efficientnet-b3')
       
        for i in range(len(test_df)):
            for j in range(-12, 134):
                test_data.append([test_df['Patient'][i],j,test_df['Age'][i],test_df['Sex'][i],test_df['SmokingStatus'][i], test_df['FVC'][i],test_df['Percent'][i]])
        test_data = pd.DataFrame(test_data, columns=['Patient','Weeks','Age','Sex','SmokingStatus','FVC','Percent'])
        
        TAB_test = {}
        Person_test = []
        for i, p in tqdm(enumerate(test_data.Patient.unique())):
            sub = test_data.loc[test_data.Patient == p]
            TAB_test[p] = get_tab(sub)
            Person_test.append(p)
        Person_test = np.array(Person_test)

        TARGET = {}
        test = test_data
        test_ds = Dataset(test_data, TAB_test,TARGET,Patient1 , filename, mode= 'test')
        test_dl = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=128,
            shuffle=True,
            collate_fn=collate_fn_test        
        )
       
        avg_predictions= np.zeros((146,1))
        for i in range(len(models)):
            predictions = []
            model = models[i]
            model = model.to(dev)
            model.load_state_dict(torch.load('/Users/saikishorehr/Documents/FYP/fold_' +str(i)+'.pth'))
            model.eval()             
            with torch.no_grad():
                for X in test_dl:
                    imgs = X[0].to(dev)
                    tabs = X[1].to(dev)
                    pred = model(imgs,tabs)
                    predictions.extend(pred.detach().cpu().numpy().tolist())
            avg_predictions += predictions

        predictions = avg_predictions / len(models)
        fvc = []
        percent=[]
        ids=[]
        week=[]
        for i in range(len(test_data)):
            p =test_data['Patient'][i]
            good_fvc=(test_df.loc[test_df.Patient==p]['FVC']*100/(test_df.loc[test_df.Patient==p]['Percent']))
            B_test = predictions[i][0] * test_df.Weeks.values[test_df.Patient == p][0]
            cur_fvc=predictions[i][0] * test_data['Weeks'][i] + test_data['FVC'][i] - B_test
            ids.append(p)
            week.append(test_data['Weeks'][i])
            fvc.append(predictions[i][0] * test_data['Weeks'][i] + test_data['FVC'][i] - B_test)
            percent.append((cur_fvc*100)/good_fvc)
        
        if not fvc:
          value="not defined"
        else:
          value=str(round(fvc[PWeeks1],2))+"    Relative Percent : "+str(round(percent[PWeeks1],2)[0]).lstrip('0')
        
        our_result=pd.DataFrame()
        our_result['Patient_ID']=ids
        our_result['FVC']=fvc
        our_result['Percent']=percent
        our_result['Week']=week
        html=our_result.to_html()
        
        text_file=open("/Users/saikishorehr/Documents/FYP/PulmonaryFibrosis/home/templates/Disease_Progression.html","w")
        text_file.write(html)
        text_file.close()

    return render(request,
                  'pulmonaryfibrosis.html',
                  {
                      'context': value,
                      
                  }
                  )
@login_required  
def getprogression(request):
    return render(request,'Disease_Progression.html')

