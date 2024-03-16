import pickle
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

loaded_model = GPT2LMHeadModel.from_pretrained("test_model_2")


input_text = f"I've been feeling so sad and overwhelmed lately. Work has become such a massive source of stress for me. {tokenizer.eos_token}"

print(input_text)

# Tokenize the input text
input_ids = tokenizer(input_text, return_attention_mask=True, return_tensors="pt")

response = torch.randint(low=10, high=5000, size=(1,50))[0]
attention = torch.zeros(50)
inputs = torch.cat((input_ids["input_ids"][0],response))


attention_mask = torch.cat((input_ids["attention_mask"][0],attention))


print(inputs)
print(attention_mask)


output = tokenizer.decode(torch.argmax(F.softmax(loaded_model(input_ids= inputs, attention_mask=attention_mask).logits, dim=1),dim=1))

print(output)