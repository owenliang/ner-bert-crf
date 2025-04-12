from dataset import NerDataset
from model import NerBertCrfModel
from modelscope import AutoTokenizer
import torch 
import functools
import os 
import tqdm

device='cuda' if torch.cuda.is_available() else 'cpu'

labels={'COUNTRY-B':0,'COUNTRY-I':1,'PERSON-B':2,'PERSON-I':3,'O':4}

def collate_fn(batch,tokenizer):
    bos_id=tokenizer.convert_tokens_to_ids(['[CLS]'])
    eos_id=tokenizer.convert_tokens_to_ids(['[SEP]'])
    pad_id=tokenizer.convert_tokens_to_ids(['[PAD]'])
    batch_x,batch_y,batch_attn_mask,batch_token_type_ids=[],[],[],[]
    max_seq_len=0
    for sample in batch:
        x=bos_id+sample[0]+eos_id
        batch_x.append(x)
        if len(x)>max_seq_len:
            max_seq_len=len(x)
        y=[labels['O']]+[labels[label] for label in sample[1]]+[labels['O']]
        batch_y.append(y)
    # padding
    for i,x in enumerate(batch_x):
        batch_attn_mask.append([1]*len(x)+[0]*(max_seq_len-len(x)))
        batch_y[i].extend([labels['O']]*(max_seq_len-len(x)))
        x.extend(pad_id*(max_seq_len-len(x))) 
        batch_token_type_ids.append([0]*len(x)) # sentence A
    return torch.tensor(batch_x,dtype=torch.long),torch.tensor(batch_attn_mask,dtype=torch.bool),torch.tensor(batch_token_type_ids,dtype=torch.long),torch.tensor(batch_y,dtype=torch.long)

if __name__=='__main__':
    tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    model=NerBertCrfModel(num_tags=len(labels)).to(device)
    try:
        model.load_state_dict(torch.load('model.pt'))
    except:
        pass
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=6e-5)

    dataset=NerDataset('data/train_label.json',tokenizer)
    dataloader=torch.utils.data.DataLoader(dataset,
                                        batch_size=16,
                                        shuffle=True,
                                        persistent_workers=True,
                                        num_workers=2,
                                        collate_fn=functools.partial(collate_fn,tokenizer=tokenizer))

    steps=0
    epoch=0
    model.train()
    while True:
        pbar=tqdm.tqdm(dataloader,total=len(dataloader),ncols=100)
        for batch_x,batch_attn_mask,batch_token_type_ids,batch_y in pbar:
            batch_x,batch_attn_mask,batch_token_type_ids,batch_y=batch_x.to(device),batch_attn_mask.to(device),batch_token_type_ids.to(device),batch_y.to(device)
            loss=model(batch_x,batch_attn_mask,batch_token_type_ids,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            steps+=1
            pbar.set_description(f'Epoch {epoch}')
            pbar.set_postfix(loss=loss.item(),steps=steps)
            
            # checkpoint
            if steps%300==0:
                torch.save(model.state_dict(),'.model.pt')
                os.replace('.model.pt','model.pt')
        epoch+=1