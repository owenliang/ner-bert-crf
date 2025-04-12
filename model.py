import torch 
from torchcrf import CRF
from modelscope import BertModel

class NerBertCrfModel(torch.nn.Module):
    def __init__(self,num_tags):
        super().__init__()
        self.bert=BertModel.from_pretrained('google-bert/bert-base-chinese')
        self.hidden=torch.nn.Linear(in_features=768,out_features=num_tags)
        self.crf=CRF(num_tags=num_tags,batch_first=True)
        for p in self.bert.parameters():
            p.requires_grad=False
    def forward(self,input_ids,attention_mask,token_type_ids,labels):
        emissions=self.bert(input_ids,attention_mask,token_type_ids)
        emissions=self.hidden(emissions.last_hidden_state)
        return -self.crf(emissions=emissions,tags=labels,mask=attention_mask) # loss 
    
    def predict(self,input_ids,attention_mask,token_type_ids):
        emissions=self.bert(input_ids,attention_mask,token_type_ids)
        emissions=self.hidden(emissions.last_hidden_state)
        return self.crf.decode(emissions=emissions,mask=attention_mask)

if __name__=='__main__':
    model=NerBertCrfModel(num_tags=5)
    input_ids=torch.randint(0,100,size=(2,20))
    attention_mask=torch.ones_like(input_ids).bool()
    token_type_ids=torch.zeros_like(input_ids)
    labels=torch.randint(0,5,size=(2,20))
    loss=model(input_ids,attention_mask,token_type_ids,labels)