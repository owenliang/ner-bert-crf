import torch
import json

class NerDataset(torch.utils.data.Dataset):
    def __init__(self,filename,tokenizer): 
        self.tokenizer=tokenizer
        with open(filename,'r',encoding='utf-8') as f:
            self.tokens_arr,self.labels_arr=self._preprocess(json.load(f))
    def _preprocess(self,data):
        tokens_arr,labels_arr=[],[]
        for item in data:
            text=item['data']['text']
            annotations=item['annotations']
            
            ch_tokens_list=[]
            ch_labels_list=[]
            for ch in text: # tokenizer by char
                ch_tokens=self.tokenizer([ch],add_special_tokens=False)
                assert len(ch_tokens.input_ids[0])<=1
                ch_tokens_list.append(ch_tokens.input_ids[0])
                ch_labels_list.append(['O']*len(ch_tokens.input_ids[0]))
            for anno in annotations[0]['result']:
                anno=anno['value']
                start,end,text,label=anno['start'],anno['end'],anno['text'],anno['labels'][0]
                first=True
                for i in range(start,end):
                    for j in range(len(ch_labels_list[i])):
                        if first:
                            ch_labels_list[i][j]=label+'-B'
                            first=False
                        else:
                            ch_labels_list[i][j]=label+'-I'
            item_tokens=[]
            item_labels=[]
            for i in range(len(ch_tokens_list)):
                item_tokens.extend(ch_tokens_list[i])
                item_labels.extend(ch_labels_list[i])
            tokens_arr.append(item_tokens)
            labels_arr.append(item_labels)
        return tokens_arr,labels_arr
            
    def __getitem__(self,idx):
        return self.tokens_arr[idx],self.labels_arr[idx]
    
    def __len__(self):
        return len(self.tokens_arr)

if __name__=='__main__':
    from modelscope import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    ds=NerDataset('data/train_label.json',tokenizer)
    tokens,labels=ds[20]
    print(tokens,len(tokens))
    print(tokenizer.decode(tokens))
    print(labels,len(labels))