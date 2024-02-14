import json
from functools import partial
from pathlib import Path
import os


import numpy as np
import pandas as pd
import torch
import transformers
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

import datasets
import argparse

from run_clm import preprocess_batch, load_training_dataset, DataCollatorForCompletionOnlyLM

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, type=str, nargs="+")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--data", required=True, type=str)
parser.add_argument("--no_input", action="store_true")
args = parser.parse_args()

# Load the pre-trained model


raw_dataset = load_training_dataset(args.data, small=False)
raw_dataset = raw_dataset.filter(lambda x: x["problem"] != "unknown")

model_names = args.models
for model_name in tqdm(model_names):
    out_path = f"scores/{model_name.replace('/', '_')}.csv"
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping")
        continue
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs("scores/", exist_ok=True)

    def preprocess_function_conditional(examples):
        inputs = [i.strip() for i in examples["input"]]
        targets = [i.strip() for i in examples["target"]]
        # inputs = [prefix + inp for inp in inputs]

        if args.no_input:
            inputs = ["N/A" for _ in inputs]

        model_inputs = tokenizer(inputs, max_length=1024, padding=False, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=256, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    device = "cuda"
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch.bfloat16)
        preprocess_function = preprocess_function_conditional
    except ValueError:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16)
        preprocess_function = partial(preprocess_batch, tokenizer=tokenizer, max_length=1024)

    train_dataset = raw_dataset.map(preprocess_function, batched=True, num_proc=1, desc="Running tokenizer on dataset")
    if 'Unnamed: 0' in train_dataset.column_names :
        if 'id'  not in train_dataset.column_names:
            train_dataset = train_dataset.rename_column('Unnamed: 0', 'id')
        else:
            train_dataset = train_dataset.remove_columns(['Unnamed: 0'])

    input_dataset = train_dataset.remove_columns([i for i in train_dataset.column_names if i not in ["input_ids",
                                                                                                         "attention_mask",
                                                                                                         "input_lengths",
                                                                                                         "labels"]])
    if "CausalLM" in type(model).__name__:
        dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=1, shuffle=False,
        collate_fn=DataCollatorForCompletionOnlyLM(tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8))
    else:
        dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=args.batch_size, shuffle=False,
                                                   collate_fn=DataCollatorForSeq2Seq(tokenizer))

    with torch.no_grad():
        model.eval()
        aum_scores = []
        ppl_scores = []
        p_mean_scores = []
        p_min_scores = []
        p_scores = []

        model.to(device)
        for batch in tqdm(dataloader):
            if "input_lengths" in batch:
                input_lengths = batch.pop("input_lengths")
            else:
                input_lengths = None
            for key in batch:
                batch[key] = batch[key].to(device)
                if batch[key].dtype == torch.float32:
                    batch[key] = batch[key].to(torch.bfloat16)
            logits = model(**batch)["logits"].to(torch.float32)
            labels = batch["labels"]
            logits[labels == -100] = -10000
            probs = torch.softmax(logits, dim=2)

            for i, labels in enumerate(labels):
                # Filter out examples with -100 label
                filtered_inputs = batch["input_ids"][i][batch["input_ids"][i] != 0]
                filtered_labels = labels[labels != -100]
                filtered_probs = probs[i][labels != -100]
                filtered_logits = logits[i][labels != -100]

                if input_lengths:
                    filtered_probs = filtered_probs[:input_lengths[i]]
                    filtered_logits = filtered_logits[:input_lengths[i]]
                    filtered_labels = filtered_labels[:input_lengths[i]]


                # Calculate probability of true label
                filtered_prob_true = filtered_probs[torch.arange(len(filtered_labels)), filtered_labels].unsqueeze(1)

                # Calculate probability of other labels
                filtered_probs[torch.arange(len(filtered_labels)), filtered_labels] = 0
                filtered_prob_max_other = filtered_probs.max(dim=1).values

                # Calculate the max difference between the two
                aum_scores.append((filtered_prob_max_other - filtered_prob_true).max().item())
                ppl_scores.append(torch.exp(F.cross_entropy(filtered_logits, filtered_labels)).detach().cpu().numpy())

                # Calculate the joint probability of the sentence
                p_scores.append(torch.exp(torch.log(filtered_prob_true).sum()).item())

                # Calculate the average probability of each token
                p_mean_scores.append(filtered_prob_true.mean().item())

                # Calculate the minimum probability of each token
                p_min_scores.append(filtered_prob_true.min().item())

        df = {"id": train_dataset["id"],
              "ppl": ppl_scores,
              "aum": aum_scores,
              "p_mean": p_mean_scores,
              "p_min": p_min_scores,
              "input": train_dataset["input"],
              "target": train_dataset["target"],
              "dataset": train_dataset["dataset"],
              "problem": train_dataset["problem"],
              }


        df = pd.DataFrame(df)
        df.to_csv(out_path)

