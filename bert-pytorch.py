import torch
from transformers import BertTokenizer
import pandas as pd
import csv
import os

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_path = './Weibo_2/'

df = pd.read_csv(os.path.join(data_path,"train.tsv"), delimiter='\t')
df_dev=pd.read_csv(os.path.join(data_path,"dev.tsv"), delimiter='\t')
df_test=pd.read_csv(os.path.join(data_path,"test.tsv"), delimiter='\t')


MAX_LEN=128
tokenizer=BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-multilingual-cased',do_lower_case=False)
#提取语句并处理
#训练集部分
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def token(tokenizer, df, df_dev, df_test):
  sentencses=['[CLS] ' + sent + ' [SEP]' for sent in df.txt.values]
  labels=df.label.values
  #这里0表示不积极,1表示积极
  labels=list(map(lambda x:0 if x == 0 else 1,[x for x in labels]))
  tokenized_sents=[tokenizer.tokenize(sent) for sent in sentencses]
  #将分割后的句子转化成数字  word-->idx
  input_ids=[tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
  #做PADDING，大于128做截断，小于128做PADDING
  input_ids=pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  #建立mask
  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

#验证集部分
  dev_sentencses=['[CLS] ' + sent + ' [SEP]' for sent in df_dev.txt.values]
  dev_labels=df_dev.label.values
  dev_labels=list(map(lambda x:0 if x == 0 else 1,[x for x in dev_labels]))
  dev_tokenized_sents=[tokenizer.tokenize(sent) for sent in dev_sentencses]
  dev_input_ids=[tokenizer.convert_tokens_to_ids(sent) for sent in dev_tokenized_sents]
  dev_input_ids=pad_sequences(dev_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  dev_attention_masks = []
  for seq in dev_input_ids:
    dev_seq_mask = [float(i>0) for i in seq]
    dev_attention_masks.append(dev_seq_mask)

#测试集部分
  test_sentencses=['[CLS] ' + sent + ' [SEP]' for sent in df_test.txt.values]
  test_labels=df_test.label.values
  test_labels=list(map(lambda x:0 if x == 0 else 1,[x for x in test_labels]))
  test_tokenized_sents=[tokenizer.tokenize(sent) for sent in test_sentencses]
  test_input_ids=[tokenizer.convert_tokens_to_ids(sent) for sent in test_tokenized_sents]
  test_input_ids=pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  test_attention_masks = []
  for seq in test_input_ids:
    test_seq_mask = [float(i>0) for i in seq]
    test_attention_masks.append(test_seq_mask)
  
#构建训练集、验证集、测试集的dataloader
  train_inputs = torch.tensor(input_ids)  
  validation_inputs = torch.tensor(dev_input_ids)
  test_inputs = torch.tensor(test_input_ids)

  train_labels = torch.tensor(labels)
  validation_labels = torch.tensor(dev_labels)
  test_labels = torch.tensor(test_labels)

  train_masks = torch.tensor(attention_masks)
  validation_masks = torch.tensor(dev_attention_masks)
  test_masks = torch.tensor(test_attention_masks)

  batch_size = 8

  train_data = TensorDataset(train_inputs, train_masks, train_labels)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
  validation_sampler = SequentialSampler(validation_data)
  validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

  test_data = TensorDataset(test_inputs, test_masks, test_labels)
  test_sampler = SequentialSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

  return train_dataloader, validation_dataloader, test_dataloader

train_dataloader, validation_dataloader, test_dataloader = token(tokenizer, df, df_dev, df_test)

from transformers import BertForSequenceClassification, AdamW, BertConfig

#装载微调模型
model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='bert-base-multilingual-cased', num_labels=2)
model.to(device)
#print(model.cuda())    # changed by yuemei

#定义优化器
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)

#学习率预热，训练时先从小的学习率开始训练
from transformers import get_linear_schedule_with_warmup
# Number of training epochs (authors recommend between 2 and 4)
epochs = 3
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np
#计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#输出格式化时间
import time
import datetime
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

from tqdm import trange
#训练部分
train_loss_set = []
epochs = 3
for _ in trange(epochs, desc="Epoch"):
    t0 = time.time()
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        batch = tuple(t.to(device) for t in batch)#将数据放置在GPU上
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
        
        train_loss_set.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Chinese Train loss: {}".format(tr_loss / nb_tr_steps))
    print("Chinese  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    t0 = time.time()
    #验证集
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Chinese Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Chinese Validation took: {:}".format(format_time(time.time() - t0)))

print("Chinese Training complete!")

#在测试集上进行测试
t0 = time.time()
model.eval()

test_loss, test_accuracy = 0, 0
nb_test_steps, nb_test_examples = 0, 0

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():        
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    tmp_test_accuracy = flat_accuracy(logits, label_ids)
    test_accuracy += tmp_test_accuracy
    nb_test_steps += 1
print("Chinese Test Accuracy: {0:.4f}".format(test_accuracy/nb_test_steps))
print("Chinese  Test took: {:}".format(format_time(time.time() - t0)))

output_dir2 = './Chinese_bert_result/'
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir2)
