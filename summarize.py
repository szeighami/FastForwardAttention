# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import numpy as np
import os
import sys
import torch
from datetime import datetime
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import collections
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str,
                        default='facebook/opt-350m')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--output_len", type=int, default=128)

    args = parser.parse_args()

    hf_model_name = args.hf_model_name
    dataset_cnn = load_dataset("ccdv/cnn_dailymail", '3.0.0', cache_dir=args.cache_path)

    hf_config = vars(AutoConfig.from_pretrained(hf_model_name))
    output_len =args.output_len




    def get_batch(datapoints, tokenizer): 
        articles = []
        tokenized_articles = []
        lengths = []
        summaries = []
        for article in datapoints['article']:
            line = article+ ' TL;DR: '
            line = line.strip()
            line = line.replace(" n't", "n't")
            line = " ".join(line.split(" ")[-1000:])
            lengths.append(len(line))
            tokenized_articles.append(line)

        tokenized_articles = tokenizer(tokenized_articles, return_tensors="pt", padding=True)

        return tokenized_articles, lengths, datapoints['article'], datapoints['highlights']


    def summarize_hf(inputs, masks, model):

        with torch.no_grad():
            outputs = model.generate(inputs ,attention_mask=masks,
                                    max_new_tokens= output_len,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)

        return outputs

    os.environ["prob_thresh"] = "0"
    os.environ["ffa_thersh"] = "0"
    os.environ["max_tokens"] = str(output_len)
    os.environ["use_ffa"] = "1"

    for batch_size in [10, 5, 2, 1]:
        np.random.seed(0)
        shuffled = np.random.permutation(11490)
        index = shuffled[:(args.max_ite//batch_size)*batch_size].reshape(-1, batch_size)
            
        for use_ffa in [2]:
            if use_ffa == 1:
                os.environ["use_ffa"] = str(use_ffa) 
                os.environ["use_ffa_default_kl"] = str(0) 
            elif use_ffa == 2:
                os.environ["use_ffa"] = str(1) 
                os.environ["use_ffa_default_kl"] = str(1) 
            else:
                os.environ["use_ffa"] = str(0) 
                os.environ["use_ffa_default_kl"] = str(0) 

            metric = load_metric("rouge")


            model = AutoModelForCausalLM.from_pretrained(hf_model_name)
            if args.data_type == 'fp16':
                model.half()
            elif args.data_type == 'bf16':
                model.bfloat16()
            model.cuda()

            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token

            curr_time = 0.0
            for i in tqdm(range(len(index))):
                datapoints = dataset_cnn['test'][index[i]]


                tokenized_articles, lengths, articles, highlights = get_batch(datapoints, tokenizer)
                inputs = tokenized_articles["input_ids"].cuda()
                mask = tokenized_articles["attention_mask"].cuda()

                seq_lens = torch.min(mask, dim=-1).indices
                seq_lens[seq_lens == 0] = mask.shape[-1]
                seq_lens = seq_lens.int()
                inputs = inputs.int()

                #try:
                start_time = datetime.now()
                outputs = summarize_hf(inputs, mask, model)

                stop_time = datetime.now()
                curr_time += (stop_time - start_time).total_seconds()
                    
                batch_out_sentence = tokenizer.batch_decode(outputs.reshape(batch_size, -1), skip_special_tokens=True)
                summary = [s.split(" TL;DR:")[1] for s in batch_out_sentence]


                metric.add_batch(predictions=summary, references=datapoints['highlights'])
                #except:
                #    print('Error with datapoint : ', i)

            computed_metrics = metric.compute()
            print(f'use_ffa={use_ffa}, batch_size={batch_size} total latency: {curr_time} sec')
            for key in computed_metrics.keys():
                print(f'{key} : {computed_metrics[key].mid[2]*100}')



if __name__ == '__main__':
    main()
