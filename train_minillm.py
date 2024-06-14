import os
import json

import torch
import torch.distributed as dist

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
)

from arguments import get_args
from utils import print_args, initialize, get_tokenizer

from minillm import train, Reward


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        config=config,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
    )

    if args.peft is not None:
        if args.peft == 'lora':
            assert args.teacher_peft_path is not None
            model = PeftModel.from_pretrained(model, args.peft_path)
        else:
            raise NotImplementedError
    else:
        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])))

    model.eval()

    return model


def main():
    args = get_args()
    initialize(args)

    device = torch.cuda.current_device()

    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['gradient_clipping'] = args.clip_grad
    ds_config['steps_per_print'] = 10000000

    args.fp32 = not ds_config['fp16']['enabled']
    args.deepspeed_config = None

    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type

    teacher_model = get_teacher_model(args, device)
    tokenizer = get_tokenizer(args)

    reward = Reward(args, tokenizer, teacher_model)

    train(
        args=args,
        tokenizer=tokenizer,
        reward_fn=reward.reward_fn,
        teacher_model=teacher_model,
        ds_config=ds_config,
        prompt_data=args.prompt_data_dir,
        eval_prompt_data=args.prompt_data_dir,
        lm_data=args.lm_data_dir,
        eval_lm_data=args.lm_data_dir,
    )


if __name__ == '__main__':
    main()
