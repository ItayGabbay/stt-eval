#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import json
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from scipy.signal import resample
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import soundfile as sf
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer
from utils.clean_arabic import clean_arabic


# In[2]:


torch.cuda.device_count()


# In[3]:


train_inputs_path = "/mnt/nfs/stt_project/dataset/reupload/train/"
train_labels_path = "/mnt/nfs/dorten/cleaned_labels/train/"
train_dataset = []
errors = 0
for audio_file in os.listdir(train_inputs_path)[:100]:
    audio_data, sample_rate = sf.read(os.path.join(train_inputs_path, audio_file))
    label_file = os.path.join(train_labels_path, audio_file.split('.')[0] + ".txt")
    try:
        with open(label_file, "r", encoding="utf-8-sig") as f:
            text = f.read().strip()
            text = clean_arabic(text)

        train_dataset.append(
            {"audio_data": audio_data, "sample_rate": sample_rate, "sentence": text}
        )
    except:
        print(f"Error openning {label_file}")
        errors += 1


# In[4]:


errors


# In[5]:


len(train_dataset)


# In[6]:


train_dataset[0]


# In[7]:


test_inputs_path = "/mnt/nfs/stt_project/dataset/reupload/test/"
test_labels_path = "/mnt/nfs/stt_project/dataset/test-txt/"
test_dataset = []
errors = 0
for audio_file in os.listdir(test_inputs_path):
    audio_data, sample_rate = sf.read(os.path.join(test_inputs_path, audio_file))
    label_file = os.path.join(test_labels_path, audio_file.split('.')[0] + ".txt")
    try:
        with open(label_file, "r", encoding="utf-8-sig") as f:
            text = f.read().strip()
            text = clean_arabic(text)
        test_dataset.append(
            {"audio_data": audio_data, "sample_rate": sample_rate, "sentence": text}
        )
    except:
        print(f"Error openning {label_file}")
        errors += 1


# In[8]:


# resample data - to remove once data is already in 16k sr
for record in train_dataset + test_dataset:
    data = record['audio_data']
    origin_sr = record['sample_rate']
    expected_sr = 16000
    if origin_sr == expected_sr:
        data_resampled = data
    else:
        data_resampled = resample(data, int(len(data) * expected_sr / origin_sr), axis=0)
    record['sample_rate'] = expected_sr
    record['audio_data'] = data_resampled


# In[9]:


train_dataset[0]


# In[10]:


test_dataset[0]


# In[11]:


def extract_all_chars(record):
  all_text = " ".join(record["sentence"])
  vocab = list(set(all_text))
  return {"vocab": vocab, "all_text": [all_text]}


# In[12]:


vocab_parts = []
for record in train_dataset + test_dataset:
    vocab_parts.append(extract_all_chars(record))


# In[13]:


vocab_parts[0]


# In[14]:


vocab_list = []
for v in vocab_parts:
    vocab_list.extend(v['vocab'])

vocab_set = set(vocab_list)


# In[15]:


vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}
vocab_dict


# In[16]:


vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]


# In[17]:


vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)


# In[18]:


target_lang="ara"


# In[19]:


new_vocab_dict = {target_lang: vocab_dict}


# In[20]:


with open('vocab.json', 'w') as vocab_file:
    json.dump(new_vocab_dict, vocab_file)


# In[20]:


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", target_lang=target_lang)


# In[21]:


feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


# In[22]:


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# In[23]:


def prepare_record(record):
    datum = {}
    # compute log-Mel input features from input audio array 
    datum["input_values"] = feature_extractor(record["audio_data"], sampling_rate=record["sample_rate"]).input_values[0]

    # encode target text to label ids 
    datum["labels"] = tokenizer(record["sentence"]).input_ids
    return datum


# In[24]:


train_prepared_records = []
for record in tqdm(train_dataset):
    train_prepared_records.append(prepare_record(record))


# In[25]:


test_prepared_records = []
for record in tqdm(test_dataset):
    test_prepared_records.append(prepare_record(record))


# In[26]:


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        model_input_name = self.processor.model_input_names[0]
        input_features = [{"input_values": feature[model_input_name]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# In[27]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# In[28]:


wer_metric = evaluate.load("wer")


# In[29]:


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# In[30]:


model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/mms-1b-all",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
)


# In[31]:


model.freeze_base_model()
model.init_adapter_layers()
adapter_weights = model._get_adapters()
for param in adapter_weights.values():
    param.requires_grad = True


# In[33]:


training_args = TrainingArguments(
    output_dir="./models/mms-1b-all-adapters",  # change to a repo name of your choice
    group_by_length=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=100,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    num_train_epochs=4,
    save_steps=4,
    eval_steps=4,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=9
)


# In[34]:


trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_prepared_records,
    eval_dataset=test_prepared_records,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# In[35]:


for batch in trainer.get_train_dataloader():
    break


# In[36]:


batch['input_values'].shape


# In[37]:


data_collator =  trainer.get_train_dataloader().collate_fn
batch = data_collator([trainer.train_dataset[i] for i in range(4)])


# In[38]:


batch['input_values'].shape


# In[ ]:


torch.cuda.device_count()


# In[51]:


trainer.train_dataset = trainer.train_dataset[:4]
trainer.eval_dataset = trainer.eval_dataset[:4] 

# In[52]:


trainer.train()


# In[44]:


out = trainer.model(**batch.to("cuda:0"))


# In[45]:


out


# In[47]:


for batch in trainer.get_eval_dataloader(trainer.eval_dataset):
    break


# In[49]:


batch.keys()


# In[ ]:




