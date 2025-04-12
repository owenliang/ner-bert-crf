import torch 
from torchcrf import CRF
from model import NerBertCrfModel
from modelscope import AutoTokenizer

device='cuda' if torch.cuda.is_available() else 'cpu'
labels={'COUNTRY-B':0,'COUNTRY-I':1,'PERSON-B':2,'PERSON-I':3,'O':4}
labels_rev=dict((v,k) for k,v in labels.items())

tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
model=NerBertCrfModel(num_tags=len(labels)).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()

def ner(s):
    tokens=[]
    for ch in s:
        ret=tokenizer([ch],add_special_tokens=False)
        tokens=tokens+ret.input_ids[0]
        
    bos_id=tokenizer.convert_tokens_to_ids(['[CLS]'])
    eos_id=tokenizer.convert_tokens_to_ids(['[SEP]'])

    input_ids=torch.tensor(bos_id+tokens+eos_id,dtype=torch.long).to(device)
    attn_mask=torch.tensor([1]*len(input_ids),dtype=torch.bool).to(device)
    type_ids=torch.tensor([0]*len(input_ids),dtype=torch.long).to(device)

    pred=model.predict(input_ids.unsqueeze(0),attn_mask.unsqueeze(0),type_ids.unsqueeze(0))
    # ignore [CLS] and [SEP]
    input_ids=input_ids[1:-1]
    pred=pred[0][1:-1]
    print(pred)
    start=None
    entity=''
    ner_result=[]
    for i in range(len(pred)):
        pred_label=labels_rev[pred[i]]
        pred_label_splits=pred_label.split('-')
        pred_label_first=pred_label_splits[0]
        pred_label_second='' if len(pred_label_splits)<=1 else pred_label_splits[1]
        if start is None and pred_label_second=='B': # entiry start
            start=i
            entity=pred_label_first
        elif start is not None and (pred_label_first!=entity or (pred_label_first==entity and pred_label_second=='B')):
            entity_value=[tokenizer.convert_ids_to_tokens([id])[0] for id in input_ids[start:i]]
            ner_result.append((start,i-1,entity,''.join(entity_value)))
            if pred_label_second=='B':
                start=i
                entity=pred_label_first
            else:
                start=None
                entity=''
    if start is not None:
        entity_value=[tokenizer.convert_ids_to_tokens([id])[0] for id in input_ids[start:]]
        ner_result.append((start,len(pred)-1,entity,''.join(entity_value)))
    return ner_result

s='特朗普拉起中美关税战，中国和欧洲会面意味着什么？美方会如何做出反应？'
result=ner(s)
print(result)