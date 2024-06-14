import os
import sys
import time
import json
import multiprocessing
import numpy as np

import torch

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from arguments import get_args
from data_utils.indexed_dataset import make_builder


# 1. Implement an Encoder, which gives it a line of input data nad it returns you the tokenized result.
class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        if line.get('input') is None or line['input'] == '':
            user_content = f'{line["instruction"]}'
        else:
            user_content = f'{line["instruction"]}\n\n{line["input"]}'

        if line.get('system') is None or line['system'] == '':
            chat = [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': line['output']},
            ]
        else:
            chat = [
                {'role': 'system', 'content': line['system']},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': line['output']},
            ]

        full_prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
        prompt = self.tokenizer.apply_chat_template(chat[:-1], add_generation_prompt=True, tokenize=False)
        prompt_tokens = self.tokenizer.encode(prompt)
        full_tokens = self.tokenizer.encode(full_prompt) + [self.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]

        if len(prompt_tokens) > self.args.max_prompt_length:
            return None, None, None, None, len(line)

        return line, prompt, prompt_tokens, response_tokens, len(line)


def main():
    print('OK')
    args = get_args()

    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)

    # with open(os.path.join(args.data_dir, "raw.jsonl")) as f:
    #     raw_data = f.readlines()

    raw_data = load_dataset(args.data_dir)
    raw_data = raw_data["train"].to_list()

    if args.dev_num > 0:
        train_data, valid_data = train_test_split(raw_data,
                                                  test_size=args.dev_num, random_state=args.seed, shuffle=True)
        all_data = {
            "train": train_data,
            "valid": valid_data,
        }
    else:
        all_data = {
            "train": raw_data
        }

    for split in all_data:
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0

        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#" * 10, split, "#" * 10)

        prompt_lens = []
        response_lens = []

        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")

        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue

            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

            json_file.write(json.dumps({
                "instruction": line["instruction"],
                "prompt": prompt_str,
                "input": line["input"],
                "output": line["output"],
            }, indent=2, ensure_ascii=False) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                      f"({lid / elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()

        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:",
              np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()
