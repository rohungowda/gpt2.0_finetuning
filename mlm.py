# or iterative mask generation where we go from prompt + mask and so on. then predict iteratively?

import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, random_split, DataLoader
from rouge_score import rouge_scorer
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re
import tracemalloc
import time
import pickle
import numpy as np

# pad autoregressive sequences to be in batches and the moment we see a pad we don't consider the loss for that specific token

'''
input - 10 123 2334 <MASK> <PAD> <PAD>,   index - 3 , the model's predicted probability,    attention_mask - 1 1 1 0 0 0   ,  actual - 22, ignore padding calculation in loss
'''

#----------------------------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
eos_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
#model.to('cuda')
special_tokens_dict = {'bos_token': '<BOS>', 'mask_token': '<MASK>', 'pad_token': '<PAD>','sep_token':'<SEP>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.pad_token
model.resize_token_embeddings(len(tokenizer))



#----------------------------------------------------------------------------------------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def rouge_calculations(actual, generated):
    rouge_scores = []
    for ref_summary, gen_summary in zip(actual, generated):
        rouge_scores.append(scorer.score(ref_summary, gen_summary))
    rouge_lis = ([rouge["rouge1"] for rouge in rouge_scores],[rouge["rouge2"] for rouge in rouge_scores] ,[rouge["rougeL"] for rouge in rouge_scores])
    results = {}
    results["rouge1"] = {"precision":np.mean([rouge.precision for rouge in rouge_lis[0]]),
                      "recall":np.mean([rouge.recall for rouge in rouge_lis[0]]),
                      "fmeasure":np.mean([rouge.fmeasure for rouge in rouge_lis[0]])}

    results["rouge2"] = {"precision":np.mean([rouge.precision for rouge in rouge_lis[1]]),
                      "recall":np.mean([rouge.recall for rouge in rouge_lis[1]]),
                      "fmeasure":np.mean([rouge.fmeasure for rouge in rouge_lis[1]])}

    results["rougeL"] = {"precision":np.mean([rouge.precision for rouge in rouge_lis[2]]),
                      "recall":np.mean([rouge.recall for rouge in rouge_lis[2]]),
                      "fmeasure":np.mean([rouge.fmeasure for rouge in rouge_lis[2]])}

    return results

def generate_mask(tensor):
    attention_mask = [0] * len(tensor)
    for i in range(len(tensor)):
        attention_mask[i] = 1
        if tensor[i] == tokenizer.sep_token_id:
            break
        
    return torch.tensor(attention_mask)



def generate_response_mask(tensor):
    attention_mask = [tokenizer.mask_token_id] * len(tensor)
    for i in range(len(tensor)):
        attention_mask[i] = tensor[i].item()
        if tensor[i].item() == tokenizer.sep_token_id:
            break
    return torch.tensor(attention_mask)

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, responses, mask_indexes):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.responses = responses
        self.mask_indexes = mask_indexes
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.responses[idx], self.mask_indexes[idx]

#----------------------------------------------------------------------------------------

file_path = 'train.csv'
df = pd.read_csv(file_path)
df = df.sample(n=1) # 500 try 1000
#----------------------------------------------------------------------------------------
prompts = []
responses = []
mask_indexes = []

for index in range(len(df)):
    conversation = df.iloc[index]['conversations']
    conversation = conversation.replace("\n", "").replace('"',"").replace("Alex","Bhumi").replace("Charlie","user").split("} {")

    for i in range(len(conversation)):
        match = re.search("'value':", conversation[i])
        conversation[i] = conversation[i][match.end():].strip()
        conversation[i] = re.sub("^\s*'|\s*'$", "", conversation[i])
    conversation[len(conversation) - 1] = conversation[len(conversation) - 1][:-2]



    for i in range(0,len(conversation) - 1,2):
        response_tokens = tokenizer.encode(f"{conversation[i+1]} {eos_token}",return_tensors="pt")[0]
        prompt_tokens = tokenizer.encode(f"{tokenizer.bos_token} {conversation[i]} {tokenizer.sep_token}",return_tensors="pt")[0]
        prompt_length = len(prompt_tokens)
        for j,resp in enumerate(response_tokens):
            prompt_tokens = torch.cat((prompt_tokens, torch.tensor([tokenizer.mask_token_id])))
            prompts.append(prompt_tokens.clone())
            responses.append(resp)
            mask_indexes.append(prompt_length)
            prompt_tokens[prompt_length] = resp
            prompt_length += 1

#----------------------------------------------------------------------------------------
padded_tensor = pad_sequence(prompts, batch_first=True, padding_value=tokenizer.pad_token_id)
attention_mask = torch.stack([(torch.logical_and(tensor != tokenizer.mask_token_id, tensor != tokenizer.pad_token_id)).type(torch.long) for tensor in padded_tensor])
#----------------------------------------------------------------------------------------

data_object = TextDataset(padded_tensor, attention_mask, responses, mask_indexes)


total_length = len(data_object)

train_length = int(0.70 * total_length)
val_length = int(0.20 * total_length)
test_length = total_length - train_length - val_length

print(f"Train length: {train_length}")
print(f"Val length: {val_length}")
print(f"Test length: {test_length}")

train_dataset, val_dataset, test_dataset = random_split(data_object, [train_length, val_length, test_length])

batch_size = 32
Epochs = 2

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#----------------------------------------------------------------------------------------

training_losses = []
validation_losses = []
testing_losses = []

#----------------------------------------------------------------------------------------


start_time = time.time()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
step_size = 3

for epoch in range(Epochs):

    print(f"Epoch: {epoch}")
    print("-------------------------training-------------------------")
    model.train()
    total_loss = 0
    len_data = len(train_dataloader)
    actual = []
    generated = []

    print(len_data)
    for i, batch in enumerate(train_dataloader):
        input_ids_b, attention_ids_b, actual_ids_b, mask_indexes_b = batch[0],batch[1], batch[2], batch[3]
        
        # forward pass
        logits = model(input_ids= input_ids_b, attention_mask=attention_ids_b).logits
        loss_logits = torch.stack([logits[j][mask_indexes_b[j]] for j in range(len(mask_indexes_b))])
        loss = loss_fn(loss_logits,actual_ids_b)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (len_data // step_size) == 0:
            print(loss.item())
            print("---------------------Batch: "+ str(i) + " ------------------------")


        


    average_loss = total_loss/len_data
    training_losses.append(average_loss)
    print(f"Average Training Loss: {average_loss}")



    print()
    print("-------------------------validation-------------------------")
    model.eval()
    total_loss = 0
    len_data = len(val_dataloader)
    actual = []
    generated = []
    print(len_data)
    with torch.no_grad():
      for i, batch in enumerate(val_dataloader):
        input_ids_b, attention_ids_b, actual_ids_b, mask_indexes_b = batch[0],batch[1], batch[2], batch[3]
        
        # forward pass
        logits = model(input_ids= input_ids_b, attention_mask=attention_ids_b).logits
        loss_logits = torch.stack([logits[j][mask_indexes_b[j]] for j in range(len(mask_indexes_b))])
        loss = loss_fn(loss_logits,actual_ids_b)
        total_loss += loss.item()



        if i % (len_data // step_size) == 0:
            print(loss.item())
            print("---------------------Batch: "+ str(i) + " ------------------------")


            # rouge calculations

            


    average_loss = total_loss/len_data
    validation_losses.append(average_loss)
    print(f"Average Validation Loss: {average_loss}")





    print()
    print("-------------------------testing-------------------------")
    total_loss = 0
    len_data = len(test_dataloader)
    actual = []
    generated = []
    print(len_data)
    with torch.no_grad():
      for i, batch in enumerate(test_dataloader):
        input_ids_b, attention_ids_b, actual_ids_b, mask_indexes_b = batch[0],batch[1], batch[2], batch[3]
        
        # forward pass
        logits = model(input_ids= input_ids_b, attention_mask=attention_ids_b).logits
        loss_logits = torch.stack([logits[j][mask_indexes_b[j]] for j in range(len(mask_indexes_b))])
        loss = loss_fn(loss_logits,actual_ids_b)
        total_loss += loss.item()


        if i % (len_data // step_size) == 0:
            print(loss.item())
            print("---------------------Batch: "+ str(i) + " ------------------------")


    average_loss = total_loss/len_data
    testing_losses.append(average_loss)
    print(f"Average Testing Loss: {average_loss}")


    model.save_pretrained(f"saved_models/test_model_{epoch}")

    print()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print()

#----------------------------------------------------------------------------------------

epoch_ = [i for i in range(Epochs)]


fig, ax = plt.subplots()

ax.plot(epoch_, training_losses, label='training')
ax.plot(epoch_, validation_losses,label='validation')
ax.plot(epoch_, testing_losses, label='testing')
ax.set_title('Loss Values')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig('Losses.png')
