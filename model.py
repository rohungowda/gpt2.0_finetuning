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

# Assuming your model is named 'model'
start_time = time.time()

# Your code here



tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

'''
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({ 'sep_token': '[SEP]'})
tokenizer.add_special_tokens({ 'bos_token': '[BOS]'})
'''

tokenizer.pad_token = tokenizer.eos_token
pad_token = tokenizer.pad_token
eos_token = tokenizer.eos_token




def generate_mask(tensor):
    attention_mask = [0] * len(tensor)
    for i in range(len(tensor)):
        attention_mask[i] = 1
        if tensor[i] == tokenizer.eos_token_id:
            attention_mask[i] = 1
            break
    return torch.tensor(attention_mask)
        

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, input_text):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.input_text = input_text

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.input_text[idx]


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')


print("loading dataset...")
# read the data and understand it
df = pd.read_csv('train.csv')
df = df.sample(n=25) # 7000
print("finished loading dataset...")

max_length = 0
inputs = []
conversation_ids = []
Y = []
print("Preprocessing dataset ...")

for index in range(len(df)):
    conversation = df.iloc[index]['conversations']
    conversation = conversation.replace("\n", "")
    conversation = conversation.replace('"',"")
    conversation = conversation.replace("Alex","Bhumi") # bot
    conversation = conversation.replace("Charlie","user") # user
    conversation = conversation.split("} {")



    for i in range(len(conversation)):
        match = re.search("'value':", conversation[i])
        conversation[i] = conversation[i][match.end():]
        conversation[i] = conversation[i].strip()
        conversation[i] = re.sub("^\s*'|\s*'$", "", conversation[i])
    conversation[len(conversation) - 1] = conversation[len(conversation) - 1][:-2]


    for i in range(0,len(conversation) - 1,2):
        inputs.append(f"{conversation[i]} {eos_token} {conversation[i+1]} {eos_token}")
        conversation_ids.append(index)

    conversation = None
    del conversation
print("Finished Preprocessing dataset ...")

print("Tokenizing and padding dataset...")
input_tokens = tokenizer(inputs, padding=True, return_tensors="pt", return_attention_mask = False)

attention_masks = [generate_mask(tensor) for tensor in input_tokens["input_ids"]]


print("Finished Tokenizing and padding dataset...")


data_object = TextDataset(input_tokens["input_ids"], attention_masks, inputs)


total_length = len(data_object)

print(total_length)

train_length = int(0.70 * total_length)
val_length = int(0.20 * total_length)
test_length = total_length - train_length - val_length


train_dataset, val_dataset, test_dataset = random_split(data_object, [train_length, val_length, test_length])

batch_size = 64
Epochs = 3

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# function variables
# need to monitor memory and loss

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

early_stopper = EarlyStopping(patience=3, min_delta=0.025)
loss_fn = torch.nn.CrossEntropyLoss()
model = GPT2LMHeadModel.from_pretrained("gpt2")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ROGUE scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
def rouge_calculations(actual, generated):
    rouge_scores = scorer.score(actual,generated)
    return rouge_scores     

earlyStopping = False

tracemalloc.start()
# move to google cloab tommorow.
for epoch in range(Epochs):
    print(tracemalloc.get_traced_memory())
    if earlyStopping:
        print(f"early stopping at {epoch}")
        break

    print(f"Epoch: {epoch}")
    print("-------------------------training-------------------------")
    model.train()
    total_loss = 0
    len_data = len(train_dataloader)

    actual = []
    generated = []

    for i, batch in enumerate(train_dataloader):
        input_ids, attention_ids, input_text = batch[0],batch[1], batch[2]
        

        # forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_ids).logits
        loss = loss_fn(logits.reshape(-1, logits.shape[2]), input_ids.reshape(-1))
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # rouge calculations
        #responses = torch.argmax(F.softmax(logits, dim=1),dim=2)
        #responses = [tokenizer.decode(seq) for seq in responses]
        #actual.extend(re.sub(r"<\|endoftext\|>", "", text) for text in input_text)
        #generated.extend(re.sub(r"<\|endoftext\|>", "", text) for text in responses)


        if i % ((len_data - 1)//1000)== 0:
            print(f"Current Loss: {loss.item()}")


    average_loss = total_loss/len_data
    training_losses.append(average_loss)
    print(f"Average Training Loss: {average_loss}")

    #scores = rouge_calculations(" ".join(actual), " ".join(generated))
    '''
        for metric, score in scores.items():
        training_rouge_scores[metric]["precision"].append(score.precision)
        training_rouge_scores[metric]["recall"].append(score.recall)
        training_rouge_scores[metric]["fmeasure"].append(score.fmeasure)
        print(f"{metric}: {score.precision:.4f}, {score.recall:.4f}, {score.fmeasure:.4f}")
    '''


    print()
    print("-------------------------validation-------------------------")
    model.eval()
    total_loss = 0
    len_data = len(val_dataloader)


    for i, batch in enumerate(val_dataloader):
        input_ids, attention_ids, input_text = batch[0],batch[1], batch[2]
        
        # forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_ids).logits
        loss = loss_fn(logits.reshape(-1, logits.shape[2]), input_ids.reshape(-1))
        total_loss += loss.item()

        # rouge calculations
        #responses = torch.argmax(F.softmax(logits, dim=1),dim=2)
        #responses = [tokenizer.decode(seq) for seq in responses]
        #actual.extend(re.sub(r"<\|endoftext\|>", "", text) for text in input_text)
        #generated.extend(re.sub(r"<\|endoftext\|>", "", text) for text in responses)


        if i % ((len_data - 1)//1000)== 0:
            print(f"Current Loss: {loss.item()}")

    average_loss = total_loss/len_data
    validation_losses.append(average_loss)
    print(f"Average Validation Loss: {average_loss}")

    '''
        scores = rouge_calculations(" ".join(actual), " ".join(generated))
    for metric, score in scores.items():
        validation_rouge_scores[metric]["precision"].append(score.precision)
        validation_rouge_scores[metric]["recall"].append(score.recall)
        validation_rouge_scores[metric]["fmeasure"].append(score.fmeasure)
        print(f"{metric}: {score.precision:.4f}, {score.recall:.4f}, {score.fmeasure:.4f}")
    '''


    if(early_stopper(average_loss)):
        earlyStopping = True


    print()
    print("-------------------------testing-------------------------")
    model.eval()
    total_loss = 0
    len_data = len(test_dataloader)



    for i, batch in enumerate(test_dataloader):
        input_ids, attention_ids, input_text = batch[0],batch[1], batch[2]
        
        # forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_ids).logits
        loss = loss_fn(logits.reshape(-1, logits.shape[2]), input_ids.reshape(-1))
        total_loss += loss.item()


        # rouge calculations
        #responses = torch.argmax(F.softmax(logits, dim=1),dim=2)
        #responses = [tokenizer.decode(seq) for seq in responses]
        #actual.extend(re.sub(r"<\|endoftext\|>", "", text) for text in input_text)
        #generated.extend(re.sub(r"<\|endoftext\|>", "", text) for text in responses)


        if i % ((len_data - 1)//1000)== 0:
            print(f"Current Loss: {loss.item()}")

    average_loss = total_loss/len_data
    testing_losses.append(average_loss)
    print(f"Average Testing Loss: {average_loss}")
    '''
        scores = rouge_calculations(" ".join(actual), " ".join(generated))
    for metric, score in scores.items():
        testing_rouge_scores[metric]["precision"].append(score.precision)
        testing_rouge_scores[metric]["recall"].append(score.recall)
        testing_rouge_scores[metric]["fmeasure"].append(score.fmeasure)
        print(f"{metric}: {score.precision:.4f}, {score.recall:.4f}, {score.fmeasure:.4f}")
    '''


    with open(f"model_{epoch}.pkl", 'wb') as f:
        pickle.dump(model, f)



    print()
    print()
tracemalloc.stop()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print()

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

#----------------------------------------------
# need to this on google colab

exit(0)

fig, ax = plt.subplots()
ax.plot(epoch_, training_rouge_scores["rouge1"]["precision"], label='precision')
ax.plot(epoch_, training_rouge_scores["rouge1"]["recall"], label='recall')
ax.plot(epoch_, training_rouge_scores["rouge1"]["fmeasure"], label='fmeasure')
ax.set_title('training_rouge1 Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('training_rogue1_scores.png')


fig, ax = plt.subplots()
ax.plot(epoch_, training_rouge_scores["rouge2"]["precision"], label='precision')
ax.plot(epoch_, training_rouge_scores["rouge2"]["recall"], label='recall')
ax.plot(epoch_, training_rouge_scores["rouge2"]["fmeasure"], label='fmeasure')
ax.set_title('training_rouge2 Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('training_rogue2_scores.png')


fig, ax = plt.subplots()
ax.plot(epoch_, training_rouge_scores["rougeL"]["precision"], label='precision')
ax.plot(epoch_, training_rouge_scores["rougeL"]["recall"], label='recall')
ax.plot(epoch_, training_rouge_scores["rougeL"]["fmeasure"], label='fmeasure')
ax.set_title('training_rougeL Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('training_rogueL_scores.png')


#-------------------------------------------


fig, ax = plt.subplots()
ax.plot(epoch_, validation_rouge_scores["rouge1"]["precision"], label='precision')
ax.plot(epoch_, validation_rouge_scores["rouge1"]["recall"], label='recall')
ax.plot(epoch_, validation_rouge_scores["rouge1"]["fmeasure"], label='fmeasure')
ax.set_title('validation_rouge1 Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('validation_rogue1_scores.png')


fig, ax = plt.subplots()
ax.plot(epoch_, validation_rouge_scores["rouge2"]["precision"], label='precision')
ax.plot(epoch_, validation_rouge_scores["rouge2"]["recall"], label='recall')
ax.plot(epoch_, validation_rouge_scores["rouge2"]["fmeasure"], label='fmeasure')
ax.set_title('validation_rouge2 Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('validation_rogue2_scores.png')


fig, ax = plt.subplots()
ax.plot(epoch_, validation_rouge_scores["rougeL"]["precision"], label='precision')
ax.plot(epoch_, validation_rouge_scores["rougeL"]["recall"], label='recall')
ax.plot(epoch_, validation_rouge_scores["rougeL"]["fmeasure"], label='fmeasure')
ax.set_title('validation_rougeL Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('validation_rogueL_scores.png')


#-------------------------------------------


fig, ax = plt.subplots()
ax.plot(epoch_, testing_rouge_scores["rouge1"]["precision"], label='precision')
ax.plot(epoch_, testing_rouge_scores["rouge1"]["recall"], label='recall')
ax.plot(epoch_, testing_rouge_scores["rouge1"]["fmeasure"], label='fmeasure')
ax.set_title('testing_rouge1 Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('testing_rogue1_scores.png')


fig, ax = plt.subplots()
ax.plot(epoch_, testing_rouge_scores["rouge2"]["precision"], label='precision')
ax.plot(epoch_, testing_rouge_scores["rouge2"]["recall"], label='recall')
ax.plot(epoch_, testing_rouge_scores["rouge2"]["fmeasure"], label='fmeasure')
ax.set_title('testing_rouge2 Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('testing_rogue2_scores.png')


fig, ax = plt.subplots()
ax.plot(epoch_, testing_rouge_scores["rougeL"]["precision"], label='precision')
ax.plot(epoch_, testing_rouge_scores["rougeL"]["recall"], label='recall')
ax.plot(epoch_, testing_rouge_scores["rougeL"]["fmeasure"], label='fmeasure')
ax.set_title('testing_rougeL Scores')
ax.set_xlabel('Epochs')
ax.set_ylabel('Scores')
ax.legend()
plt.savefig('testing_rogueL_scores.png')

