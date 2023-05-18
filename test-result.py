import torch
from transformers import BertTokenizer
import pandas as pd
import csv
import os

device='cuda' if torch.cuda.is_available() else 'cpu'

data_path = '/home/xym/ljw/bert/French/'
df_test=pd.read_csv(os.path.join(data_path,"test.tsv"), delimiter='\t')

MAX_LEN=128
tokenizer=BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-multilingual-cased',do_lower_case=False)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def tokentest(tokenizer, df):
  sentencses=['[CLS] ' + sent + ' [SEP]' for sent in df.txt.values]
  labels=df.label.values

  labels=list(map(lambda x:0 if x == 0 else 1,[x for x in labels]))
# dev_labels=[to_categorical(i, num_classes=3) for i in dev_labels]
  tokenized_sents=[tokenizer.tokenize(sent) for sent in sentencses]
  input_ids=[tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
  input_ids=pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

  inputs = torch.tensor(input_ids)
  labels = torch.tensor(labels)
  masks = torch.tensor(attention_masks)
  batch_size = 8
  # Create the DataLoader for our validation set.
  data = TensorDataset(inputs, masks, labels)
  sampler = SequentialSampler(data)
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

  return dataloader

test_dataloader = tokentest(tokenizer, df_test)

from transformers import BertForSequenceClassification, AdamW, BertConfig
model = BertForSequenceClassification.from_pretrained('/home/xym/ljw/bert/EWC0.05_English_Frenchbase_result/', num_labels=2)
model.to(device)
print(model.cuda())

import numpy as np
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#输出格式化时间
import time
import datetime
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

#test
t0 = time.time()
model.eval()
  # Tracking variables 
test_loss, test_accuracy = 0, 0
nb_test_steps, nb_test_examples = 0, 0

  # Evaluate data for one epoch
for batch in test_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids,
        token_type_ids=None, 
        attention_mask=b_input_mask)
      # Get the "logits" output by the model. The "logits" are the output
      # values prior to applying an activation function like the softmax.
    logits = outputs[0]
      # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
      # Calculate the accuracy for this batch of test sentences.
    tmp_test_accuracy = flat_accuracy(logits, label_ids)
      
      # Accumulate the total accuracy.
    test_accuracy += tmp_test_accuracy
  
      # Track the number of batches
    nb_test_steps += 1
print("EWC French Test Accuracy: {0:.4f}".format(test_accuracy/nb_test_steps))
print("EWC French  Test took: {:}".format(format_time(time.time() - t0)))
