import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, random_split, DataLoader
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re
import tracemalloc
import time
import pickle
import numpy as np

#----------------------------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
eos_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
#model.to('cuda')
special_tokens_dict = {'bos_token': '<BOS>', 'mask_token': '<MASK>', 'pad_token': '<PAD>','sep_token':'<SEP>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.pad_token
model.resize_token_embeddings(len(tokenizer))



for name, param in model.named_parameters():
    if 'wte' in name or 'wpe' in name:  # assuming 'wte' and 'wpe' correspond to embedding layers
      param.requires_grad = False
    else:
      param.requires_grad= True

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
    def __init__(self, input_ids, input_attention_mask, actual_ids, loss_attention):
        self.input_ids = input_ids
        self.input_attention_mask = input_attention_mask
        self.actual_ids = actual_ids
        self.loss_attention = loss_attention
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.input_attention_mask[idx], self.actual_ids[idx], self.loss_attention[idx]

#----------------------------------------------------------------------------------------

file_path = 'train.csv'
df = pd.read_csv(file_path)
df = df.sample(n=1) # 500 try 1000
#----------------------------------------------------------------------------------------
actual_text = []
input_text = []

for index in range(len(df)):
    conversation = df.iloc[index]['conversations']
    conversation = conversation.replace("\n", "").replace('"',"").replace("Alex","Bhumi").replace("Charlie","user").split("} {")

    for i in range(len(conversation)):
        match = re.search("'value':", conversation[i])
        conversation[i] = conversation[i][match.end():].strip()
        conversation[i] = re.sub("^\s*'|\s*'$", "", conversation[i])
    conversation[len(conversation) - 1] = conversation[len(conversation) - 1][:-2]



    for i in range(0,len(conversation) - 1,2):
        response = f"{conversation[i+1]} {eos_token}"
        prompt = f"{tokenizer.bos_token} {conversation[i]} {tokenizer.sep_token}"

        

        actual_text.append(f"{prompt}{response}")


#----------------------------------------------------------------------------------------


actual = tokenizer(actual_text, padding=True, return_tensors="pt", return_attention_mask=True)

actual_ids, loss_attention = actual["input_ids"], actual["attention_mask"]
input_ids = torch.stack([generate_response_mask(tensor.clone()) for tensor in actual_ids])
input_attention_mask = torch.stack([generate_mask(tensor.clone()) for tensor in input_ids])


#----------------------------------------------------------------------------------------
# input_ids, forward attention mask, actual_ids loss attention mask
data_object = TextDataset(input_ids, input_attention_mask, actual_ids, loss_attention) 

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

training_rouge_scores = {"rouge1":{"precision":[],"recall":[],"fmeasure":[]}, 
                "rouge2":{"precision":[],"recall":[],"fmeasure":[]},
                "rougeL":{"precision":[],"recall":[],"fmeasure":[]}}

validation_rouge_scores = {"rouge1":{"precision":[],"recall":[],"fmeasure":[]}, 
                "rouge2":{"precision":[],"recall":[],"fmeasure":[]},
                "rougeL":{"precision":[],"recall":[],"fmeasure":[]}}

testing_rouge_scores = {"rouge1":{"precision":[],"recall":[],"fmeasure":[]}, 
                "rouge2":{"precision":[],"recall":[],"fmeasure":[]},
                "rougeL":{"precision":[],"recall":[],"fmeasure":[]}}

#----------------------------------------------------------------------------------------

start_time = time.time()

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
step_size = 1

# move to google cloab tommorow.
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
        input_ids_b, attention_ids_b, actual_ids_b, loss_attention_b = batch[0],batch[1], batch[2], batch[3]
        
        # forward pass
        logits = model(input_ids= input_ids_b, attention_mask=attention_ids_b).logits


        loss = loss_fn(logits.reshape(-1, logits.size(-1)),actual_ids_b.reshape(-1))
        change = loss_attention_b.view(-1)
        loss = torch.mean(loss[change == 1])
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (len_data // step_size) == 0:
            print("Predicted: ")
            print(re.sub(r"<\|endoftext\|>", "", tokenizer.decode(torch.argmax(F.softmax(logits[0], dim=1),dim=1))))
            print("\n------------------------------------------------------\n")
            print("Actual: ")
            print(re.sub(r"<\|endoftext\|>", "", tokenizer.decode(actual_ids_b[0])))
            print("Loss: ")
            print("\n------------------------------------------------------\n")
            print(loss.item())


            prediction = torch.argmax(F.softmax(logits, dim=1),dim=2)
            prediction = [re.sub(r"<\|endoftext\|>", "", tokenizer.decode(seq)) for seq in prediction]
            actual.extend([re.sub(r"<\|endoftext\|>", "", tokenizer.decode(text)) for text in actual_ids_b])
            generated.extend(prediction)

        print("---------------------Batch: "+ str(i) + " ------------------------")


        


    average_loss = total_loss/len_data
    training_losses.append(average_loss)
    print(f"Average Training Loss: {average_loss}")

    scores = rouge_calculations(actual,generated)
    for metric, score in scores.items():
        training_rouge_scores[metric]["precision"].append(score["precision"])
        training_rouge_scores[metric]["recall"].append(score["recall"])
        training_rouge_scores[metric]["fmeasure"].append(score["fmeasure"])
        print(f"Average {metric}: {score['precision']}, {score['recall']}, {score['fmeasure']}")

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
          input_ids_b, attention_ids_b, actual_ids_b, loss_attention_b = batch[0],batch[1], batch[2], batch[3]
          

        # forward pass
          logits = model(input_ids= input_ids_b, attention_mask=attention_ids_b).logits

          loss = loss_fn(logits.reshape(-1, logits.size(-1)),actual_ids_b.reshape(-1))
          change = loss_attention_b.view(-1)
          loss = torch.mean(loss[change == 1])
          total_loss += loss.item()
          
          if i % (len_data // step_size) == 0:
            print("Predicted: ")
            print(re.sub(r"<\|endoftext\|>", "", tokenizer.decode(torch.argmax(F.softmax(logits[0], dim=1),dim=1))))
            print("\n------------------------------------------------------\n")
            print("Actual: ")
            print(re.sub(r"<\|endoftext\|>", "", tokenizer.decode(actual_ids_b[0])))
            print("Loss: ")
            print("\n------------------------------------------------------\n")
            print(loss.item())


            prediction = torch.argmax(F.softmax(logits, dim=1),dim=2)
            prediction = [re.sub(r"<\|endoftext\|>", "", tokenizer.decode(seq)) for seq in prediction]
            actual.extend([re.sub(r"<\|endoftext\|>", "", tokenizer.decode(text)) for text in actual_ids_b])
            generated.extend(prediction)

          print("---------------------Batch: "+ str(i) + " ------------------------")


            # rouge calculations

            


    average_loss = total_loss/len_data
    validation_losses.append(average_loss)
    print(f"Average Validation Loss: {average_loss}")

    scores = rouge_calculations(actual,generated)
    for metric, score in scores.items():
        validation_rouge_scores[metric]["precision"].append(score["precision"])
        validation_rouge_scores[metric]["recall"].append(score["recall"])
        validation_rouge_scores[metric]["fmeasure"].append(score["fmeasure"])
        print(f"Average {metric}: {score['precision']}, {score['recall']}, {score['fmeasure']}")




    print()
    print("-------------------------testing-------------------------")
    total_loss = 0
    len_data = len(test_dataloader)
    actual = []
    generated = []
    print(len_data)
    with torch.no_grad():
      for i, batch in enumerate(test_dataloader):
          input_ids_b, attention_ids_b, actual_ids_b, loss_attention_b = batch[0],batch[1], batch[2], batch[3]
          
        # forward pass
          logits = model(input_ids= input_ids_b, attention_mask=attention_ids_b).logits

          loss = loss_fn(logits.reshape(-1, logits.size(-1)),actual_ids_b.reshape(-1))
          change = loss_attention_b.view(-1)
          loss = torch.mean(loss[change == 1])
          total_loss += loss.item()

          if i % (len_data // step_size) == 0:
            print("Predicted: ")
            print(re.sub(r"<\|endoftext\|>", "", tokenizer.decode(torch.argmax(F.softmax(logits[0], dim=1),dim=1))))
            print("\n------------------------------------------------------\n")
            print("Actual: ")
            print(re.sub(r"<\|endoftext\|>", "", tokenizer.decode(actual_ids_b[0])))
            print("Loss: ")
            print("\n------------------------------------------------------\n")
            print(loss.item())

            prediction = torch.argmax(F.softmax(logits, dim=1),dim=2)
            prediction = [re.sub(r"<\|endoftext\|>", "", tokenizer.decode(seq)) for seq in prediction]
            actual.extend([re.sub(r"<\|endoftext\|>", "", tokenizer.decode(text)) for text in actual_ids_b])
            generated.extend(prediction)

          print("---------------------Batch: "+ str(i) + " ------------------------")


    average_loss = total_loss/len_data
    testing_losses.append(average_loss)
    print(f"Average Testing Loss: {average_loss}")

    scores = rouge_calculations(actual,generated)
    for metric, score in scores.items():
        testing_rouge_scores[metric]["precision"].append(score["precision"])
        testing_rouge_scores[metric]["recall"].append(score["recall"])
        testing_rouge_scores[metric]["fmeasure"].append(score["fmeasure"])
        print(f"Average {metric}: {score['precision']}, {score['recall']}, {score['fmeasure']}")


    # savefile /content/drive/MyDrive/deep learning/

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




for metric, score in training_rouge_scores.items():
    fig, ax = plt.subplots()
    ax.plot(epoch_, score["precision"], label='precision')
    ax.plot(epoch_, score["recall"], label='recall')
    ax.plot(epoch_, score["fmeasure"], label='fmeasure')
    ax.set_title(f"training_{metric} Scores")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Scores')
    ax.legend()
    plt.savefig(f"training_{metric}_scores.png")

for metric, score in validation_rouge_scores.items():
    fig, ax = plt.subplots()
    ax.plot(epoch_, score["precision"], label='precision')
    ax.plot(epoch_, score["recall"], label='recall')
    ax.plot(epoch_, score["fmeasure"], label='fmeasure')
    ax.set_title(f"validation_{metric} Scores")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Scores')
    ax.legend()
    plt.savefig(f"validation_{metric}_scores.png")

for metric, score in testing_rouge_scores.items():
    fig, ax = plt.subplots()
    ax.plot(epoch_, score["precision"], label='precision')
    ax.plot(epoch_, score["recall"], label='recall')
    ax.plot(epoch_, score["fmeasure"], label='fmeasure')
    ax.set_title(f"testing_{metric} Scores")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Scores')
    ax.legend()
    plt.savefig(f"testing_{metric}_scores.png")


#----------------------------------------------------------------------------------------

predict_input_text = f"{tokenizer.bos_token} Hey Bhumi recently I've had problems sleeping becasue I'm very stressed from school work. I feel like I just don't get what they are teaching at school. {tokenizer.sep_token}"
predict = tokenizer(predict_input_text, return_attention_mask=True, return_tensors="pt")
predict_input_ids, predict_attention_mask = predict["input_ids"], predict["attention_mask"]

predict_response_tokens = torch.tensor([tokenizer.mask_token_id] * 200)
predict_input_ids = predict_input_ids + predict_response_tokens + torch.tensor([tokenizer.eos_token_id])
predict_attention_mask = predict_attention_mask + torch.mask(201)



print(predict_input_ids)
print(predict_attention_mask)


predict_model = GPT2LMHeadModel.from_pretrained("saved_models/test_model_2")

predict_output = tokenizer.decode(torch.argmax(F.softmax(predict_model(input_ids= predict_input_ids, attention_mask=predict_attention_mask).logits, dim=1),dim=1))

print(predict_output)