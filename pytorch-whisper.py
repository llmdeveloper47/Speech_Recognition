#!/usr/bin/env python
# coding: utf-8

# # Next Steps
# 
#  - Divide the final_dataset into train and test ( 95/5 split)
#  - Train Whisper on Dataset and evaluate on test
#  

# In[1]:


#!aws s3 cp s3://asr-whisper/ . --recursive


# In[1]:



# !pip3 install transformers accelerate
# !pip3 install soundfile
# !pip3 install jiwer
# !pip3 install tensorboardX
# !pip3 install -U scikit-learn
# !pip3 install evaluate>=0.30
# !pip3 install gradio
# !pip3 install transformers

# !pip3 install -U accelerate deepspeed
# !pip3 install bitsandbytes



import os
from tqdm.auto import tqdm
from accelerate import Accelerator
import torch
from os import listdir
from os.path import isfile, join
import torch
import pandas as pd
import pydub
import evaluate
from pydub import playback
from datasets import Dataset, Audio,DatasetDict,load_from_disk
from collections import Counter
import torch

from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

from transformers import AdamW, get_scheduler

from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader


def load_audio_timestamp_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path).iloc[:,1:]

    columns = dataset.columns.tolist()
    dataset = dataset.reset_index()
    dataset.columns = ['Index'] + columns
    
    return dataset


def get_audio_path(audio_path):


    all_folders = []
    all_data = []
    all_folders = os.listdir(audio_path)
    all_folders.sort()
    for folder in all_folders:

        all_files_path = []

        if folder == ".ipynb_checkpoints":
            continue
        else:

            files = os.listdir(audio_path + folder + "/")
            for file in files:
                if file in ".ipynb_checkpoints" or ".txt" in file:
                    continue
                else:
                    all_files_path.append(audio_path + folder + "/" + file)
            all_files_path.sort(key = lambda x : [int(x.split("/")[-1].split(".")[0].split("-")[-1])])
            all_folders = [folder] * len(all_files_path)


            result = pd.DataFrame(list(zip(all_folders, all_files_path)), columns = ['file_name','audio_path'])
            all_data.append(result)

    final_result = pd.concat(all_data, axis = 0)
    
    final_result = final_result.reset_index()
    final_result = final_result.drop(['index'], axis = 1)
    final_result = final_result.reset_index()
    final_result.columns = ['Index','file_name','audio_path']

    return final_result

def build_audio_dataset(directory, dataset_path):
    
    dataset = load_audio_timestamp_dataset(dataset_path)
    audio_paths =  get_audio_path(directory)
    final_dataset = pd.merge(dataset, audio_paths, how = 'inner', left_on = ['Index','file_name'], right_on =['Index','file_name'])
    
    final_dataset = final_dataset.drop(['Index'], axis = 1)
    
    
    
    return final_dataset


def get_train_test_dataset(final_dataset):
    
    all_filenames = final_dataset['file_name'].unique().tolist()

    Train_files, Test_files = train_test_split(all_filenames, test_size=0.3, random_state=412)
    train_dataset_original = final_dataset[final_dataset.file_name.isin(Train_files)]
    test_dataset = final_dataset[final_dataset.file_name.isin(Test_files)]
    
    all_filenames_train = train_dataset_original['file_name'].unique().tolist()


    Train_files, Validation_files = train_test_split(all_filenames_train, test_size=0.3, random_state=412)
    train_dataset = train_dataset_original[train_dataset_original['file_name'].isin(Train_files)]
    validation_dataset = train_dataset_original[train_dataset_original['file_name'].isin(Validation_files)]

    
    
    final_audio_dataset = DatasetDict()
    
    
    return train_dataset, validation_dataset, test_dataset, final_audio_dataset

    

def generater_vocabulary(train_dataset):
    
    dataset = train_dataset.copy()
    
    counter = Counter()
    
    dataset["Sentences"] = dataset["Sentences"].apply(lambda x : counter.update(x.split()))
    
    return counter



def create_hf_datset(dataset_train, dataset_validation, dataset_test, final_audio_dataset, dataset_type):
    
    
    
    final_audio_dataset["train"] =  Dataset.from_dict({"audio":dataset_train['audio_path'], "text": dataset_train["Sentences"]}).cast_column("audio", Audio(sampling_rate = 16000))
    final_audio_dataset["validation"] =  Dataset.from_dict({"audio":dataset_validation['audio_path'], "text": dataset_validation["Sentences"]}).cast_column("audio", Audio(sampling_rate = 16000))
    final_audio_dataset["test"] =  Dataset.from_dict({"audio":dataset_test['audio_path'], "text": dataset_test["Sentences"]}).cast_column("audio", Audio(sampling_rate = 16000))

            
        
    return final_audio_dataset


def load_whisper_artifacts(checkpoint, final_audio_dataset):
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
    
    processor = WhisperProcessor.from_pretrained(checkpoint, language="English", task="transcribe")

    tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="English", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    device = torch.device('mps')
    model.to("cuda")
    
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch
    
    
    final_audio_dataset = final_audio_dataset.map(prepare_dataset, remove_columns = final_audio_dataset.column_names["train"], num_proc= 4)
    
    
    return feature_extractor, tokenizer, model, processor, final_audio_dataset
    

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
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









# import os

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"

def master_function():
    

    accelerator = Accelerator()
    bucket_name = "./data/Audio_Segments/"
    dataset_path = "./data/audio_start_end.csv" 


    final_dataset = build_audio_dataset(bucket_name, dataset_path)

    
    train_dataset, validation_dataset, test_dataset, final_audio_dataset = get_train_test_dataset(final_dataset)
    
    vocabulary = generater_vocabulary(train_dataset)
    
    
    final_audio_dataset = create_hf_datset(train_dataset, validation_dataset, test_dataset, final_audio_dataset, "train")


    
    checkpoint = "openai/whisper-small"
    feature_extractor, tokenizer, model, processor, final_audio_dataset = load_whisper_artifacts(checkpoint, final_audio_dataset)

    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)



    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    train_dataloader = DataLoader(
        final_audio_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(
        final_audio_dataset["validation"], batch_size=8, collate_fn=data_collator
    )



    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
     train_dataloader, eval_dataloader, model, optimizer)



    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=1e-5)


    lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,)

    progress_bar = tqdm(range(num_training_steps))


    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions = predictions.predictions
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric = metric.compute()
        accelerator.print(f"epoch {epoch}:", eval_metric)

    print('\n ----- Success\a')

if __name__ == "__main__":
    torch.cuda.empty_cache()
    master_function()
    
    