

from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from collections import Counter
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
tokenizer.add_special_tokens({'additional_special_tokens':['<H>','<T>','<S>']})
model = AutoModel.from_pretrained("google/muril-base-cased")
model.resize_token_embeddings(len(tokenizer))
# import wandb
from numpyencoder import NumpyEncoder

# wandb.init(project="relation_prediction", entity="pkandru")


# wandb.config = {
#           "learning_rate": 0.0001,
#             "epochs": 30,
#               "batch_size": 16,
#               "step schedule":2,
#               "step gamma":0.5,
#               }
import json
from tqdm import tqdm





trans_lang_codes={
    'en':'eng',
    'hi':'hin',
    'pa':'pan',
    'gu':'guj',
    'or':'ori',
    'as':'ben',
    'te':'tel',
    'ml':'mal',
    'mr':'hin',
    'ta':'tam',
    'bn':'ben',
    'kn':'kan'
}

ignore_relas  = ['member of political party', 'employer', 'sport', 'date of birth', 'languages spoken, written or signed', 'nominated for', 'award received', 'league', 'religion', 'occupation', 'date of death', 'residence', 'position held', 'country of citizenship', 'work period (end)', 'place of burial', 'place of death', 'mother', 'educated at', 'place of birth', 'member of sports team', 'genre', 'convicted of', 'candidacy in election', 'work period (start)', 'work location', 'student of', 'position played on team / speciality', 'child', 'military branch', 'field of work', 'member of', 'discography', 'participant in', 'victory', 'instrument', 'sibling', 'sports discipline competed in', 'father', 'ethnic group', 'cause of death', 'movement', 'military rank', 'spouse', 'number of children', 'noble title', 'part of', 'filmography', 'title of chess person', 'family', 'named after', 'notable work', 'birth name', 'nickname', 'name in native language', 'has works in the collection', 'medical condition', 'place of detention', 'relative', 'related category', 'influenced by', 'manner of death', 'academic degree', 'country for sport', 'writing language', 'conflict', 'affiliation', 'competition class', 'doctoral advisor', 'record label', 'canonization status', 'bowling style', 'owner of', 'native language', 'student', 'coach of sports team', 'depicted by', 'killed by', 'allegiance', 'honorific prefix', 'religious order', 'feast day', 'professorship', "topic's main template", 'Roman praenomen', 'time period', 'present in work']

class_lookup ={ j:i for i,j in enumerate(ignore_relas)}
rev_class_lookup ={i:j for i,j in enumerate(ignore_relas)}



def get_train_val_test(lang_code):
    finaldata = json.load(open('transformed/finaldata.json','r'))
    train,val,test = (finaldata['./'+lang_code+'/train.jsonl'],
           finaldata['./'+lang_code+'/val.jsonl'],
           finaldata['./'+lang_code+'/test.jsonl'])
    test_rel = set([fact[0] for i in test for fact in i['facts']])
    val_rel = set([fact[0] for i in val for fact in i['facts']])
    train_rel = set([fact[0] for i in train for fact in i['facts']])
    bad_relas = (test_rel-train_rel).union(val_rel-train_rel)
    if len(bad_relas)>0:
#         print(bad_relas)
        val = [i for i in val
               if not sum([1 if fact[0] in bad_relas else 0 for fact in i['facts']])]
        test = [i for i in test
               if not sum([1 if fact[0] in bad_relas else 0 for fact in i['facts']])]
        finaldata['./'+lang_code+'/val.jsonl'] = val
        finaldata['./'+lang_code+'/test.jsonl'] = test
#         print((test_rel-train_rel).union(val_rel-train_rel))
        json.dump(finaldata,open('transformed/finaldata.json','w'))
    return train,val,test
    
# for i in trans_lang_codes:
#     train,val,test = get_train_val_test(i)

import torch
from torch import nn

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,ConcatDataset

class myDataset(Dataset):
    def __init__(self,data,
                 class_lookup,
                 trans_lookup=None,
                 transliter_lookup=None,
                 items_lookup=None,
                 asitis=False):
        self.trans_lookup = trans_lookup
        self.transliter_lookup = transliter_lookup
        self.items_lookup = items_lookup
        self.asitis = asitis
        
        self.class_lookup = class_lookup
        #print('Total number of facts ',Counter([fact[0] for facts in data for fact in facts['facts']]))    
        
        #print('Total number of facts  not in ignore relas ',len([1 for facts in data for fact in facts['facts'] if fact[0] not in keep_relas]))
        self.X =['<H> '+facts['entity_name']+self.get_tails(fact[1]) +' [SEP] ' +facts['sentence']
                for facts in data for fact in facts['facts'] if fact[0] in ignore_relas and fact[-1]!=-1]
        self.language =[facts['lang'] 
                for facts in data for fact in facts['facts'] if fact[0] in ignore_relas and fact[-1]!=-1]
        self.sentence =[facts['sentence'] 
                for facts in data for fact in facts['facts'] if fact[0] in ignore_relas and fact[-1]!=-1]
        self.y = [self.class_lookup[fact[0]] 
                for facts in data for fact in facts['facts'] if fact[0] in ignore_relas and fact[-1]!=-1]
    
    def __getitem__(self,idx):
        return tokenizer.encode(self.X[idx]),self.language[idx],self.sentence[idx], self.y[idx]

    
    def __len__(self):
        return len(self.y)
        
    def get_tails(self,x):
        string =''
        if self.trans_lookup:
            string += ' <T> '+self.trans_lookup[x]
        if self.transliter_lookup:
            string += ' <T> '+self.transliter_lookup[x]
        if self.items_lookup:
            string += ' <T> '+self.items_lookup[x]
        if self.asitis:
            string += ' <T> '+x
        return string

class datasetProvider():
    def __init__(self,lang):
        self.train,self.val,self.test = get_train_val_test(lang)
        try:
            self.train_translation_lookup = json.load(open('transformed/'+lang+'_train_translation.json','r'))
            self.val_translation_lookup = json.load(open('transformed/'+lang+'_val_translation.json','r'))
            self.test_translation_lookup = json.load(open('transformed/'+lang+'_test_translation.json','r'))
        except:
            pass
        try:
            self.train_transliteration_lookup = json.load(open('transformed/'+lang+'_train_transliteration.json','r'))
            self.val_transliteration_lookup = json.load(open('transformed/'+lang+'_val_transliteration.json','r'))
            self.test_transliteration_lookup = json.load(open('transformed/'+lang+'_test_transliteration.json','r'))
        except:
            pass
#         classes = [fact[0] for facts in self.train for fact in facts['facts'] if fact[0] not in class_lookup and fact[0] in keep_relas]
#         prev_len = len(class_lookup)
# #         print('Previous Length ',prev_len)
#         curr_classes = {v:(k+prev_len) for k,v in enumerate(set(classes)) if v not in class_lookup}
#         class_lookup.update(curr_classes)
# #         print('Current Length',len(class_lookup))
        
#         rev_class_lookup ={v:k for k,v in class_lookup.items()}
        
    def provide(self,split='train',trans_lookup=True,transliter_lookup=False,items_lookup=False,asitis=False):
        return myDataset(getattr(self, split),
                         class_lookup,
                         trans_lookup = getattr(self, split+'_translation_lookup') if trans_lookup else None,
                         transliter_lookup = getattr(self, split+'_transliteration_lookup') if transliter_lookup else None,
                         items_lookup = getattr(self, split+"_items_lookup") if items_lookup else None,
                         asitis=asitis)
        

tes=[]
for i in tqdm(trans_lang_codes):
    provider  = datasetProvider(i)
    te =  provider.provide(split='test')
    print('Length of test dataset of ',i)
    print(len(te))
    tes.append(te)

    
    
import numpy as np
class MyCollator(object):
    '''
    Yields a batch from a list of Items
    Args:
    test : Set True when using with test data loader. Defaults to False
    percentile : Trim sequences by this percentile
    '''
    def __init__(self,test=False,percentile=100):
        self.test = test
        self.percentile = percentile
    def __call__(self, batch):
        if not self.test:
            data = [item[0] for item in batch]
            language = [item[1] for item in batch]
            sentence = [item[2] for item in batch]
            target = [item[3] for item in batch]
        else: 
            data = [item[0] for item in batch]
            language = [item[1] for item in batch]
            sentence = [item[2] for item in batch]
            target = [item[3] for item in batch]
        lens = [len(x) for x in data]
        max_len = np.percentile(lens,self.percentile)
        data = [i+[0]*int(max_len-len(i)) for i in data]
        data = torch.tensor(data,dtype=torch.long)
        if not self.test:
            target = torch.tensor(target,dtype=torch.long)
            return [data,language,sentence,target]
        return [data,language,sentence,target]

collate = MyCollator(percentile=100)




batch_size=16
# train_loader = DataLoader(ConcatDataset(ts), batch_size=batch_size, shuffle=True ,collate_fn=collate)
# val_loader = DataLoader(ConcatDataset(vs), batch_size=16, shuffle=True ,collate_fn=collate)
test_loader = DataLoader(ConcatDataset(tes), batch_size=16, shuffle=True ,collate_fn=collate)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

class ClassifierModel(nn.Module):
    def __init__(self,model,otpsize=2000):
        super(ClassifierModel, self).__init__()
        self.model=model
        self.projector = nn.ModuleList([nn.Linear(768,512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512,otpsize),
                                        ])
    def forward(self,X):
        o = self.model(X).pooler_output
        for layer in self.projector:
            o = layer(o)
        return o




loss_ctn = nn.CrossEntropyLoss()


def get_mets(report):
    for i in report.split('\n'):
        if i.strip().startswith('accuracy'):
            acc = [j for j in i.split() if j!=''][-2]

        if i.strip().startswith('macro avg'):
            mac = [j for j in i.split() if j!=''][-2]

        if i.strip().startswith('weighted avg'):
            wei = [j for j in i.split() if j!=''][-2]
    return float(acc),float(mac),float(wei)


def validation_loop(model,val_loader,split='validation',save=False,epoch=0):
    model.eval()
    print('Running Eval loop')
    with torch.no_grad():
        losses=[]
        preds = []
        originals=[]
        langs=[]
        sents=[]
        for step,(X,lang,sent,y) in enumerate(val_loader):
            originals.extend(y.numpy())
            X,y = X.to(device),y.to(device)
            otp = cls(X)
            loss = loss_ctn(otp,y)
            langs.extend(lang)
            sents.extend(sent)
            losses.append(loss.item())
            preds.extend(np.argmax(otp.detach().cpu().numpy(),axis=1))
    acc = sum([1 if i==j else 0 for i,j in zip(originals,preds)])/len(preds)
    print(Counter(preds).most_common(5))
    print(split+' Accuracy ',acc)
    report = classification_report(originals,preds)
    acc,mac,wei = get_mets(report)
    if save:
        json.dump({"sentence":sents,
              "langs":langs,
              "originals":originals,
              "preds":preds}
              ,open('saved_metrics/'+str(epoch)+'.json','w'),cls=NumpyEncoder)
    
    print({split+" loss":sum(losses)/len(losses),
        split+" accuracy":acc,
        split+" macro f1":mac,
       split+" weighted f1":wei})

    return sum(losses)/len(losses)



if __name__=="__main__":
    cls = ClassifierModel(model,len(class_lookup))
    cls.load_state_dict(torch.load('saved_models/9.pt')) 
    cls.to(device)
    validation_loop(cls,test_loader,"test",True,9)