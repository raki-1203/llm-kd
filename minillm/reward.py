import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


class Reward():
    def __init__(self, args, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def get_input_batch(self, input_ids, gen_ids, output_pos=True):
        full_ids = torch.cat([input_ids, gen_ids], dim=-1)
        attention_mask = (full_ids != self.pad_token_id)

        model_inputs = {
            'input_ids': full_ids,
            'attention_mask': attention_mask,
            'use_cache': False,
        }

        if (self.args.model_type in ['gpt2']) and output_pos:
            position_ids = torch.cumsum(attention_mask, dim=-1) - 1
            position_ids.masked_fill_(~attention_mask, 0)
            model_inputs['position_ids'] = position_ids

        return model_inputs

    def reward_fn(self, input_ids, gen_ids, inf_mask=None, output_pos=True):
        # not include eos token

        self.model.eval()
        # input_ids = input_ids.repeat(1, 1)

        model_inputs = self.get_input_batch(input_ids, gen_ids, output_pos=output_pos)

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        logits = outputs.logits  # (B, L, V)
        logits = logits - torch.mean(logits, dim=-1, keepdim=True)

        mask = model_inputs['attention_mask']
        logits = logits * mask.unsqueeze(-1)  # set logits output by padding to 0

        logits = logits[:, input_ids.size(-1)-1:, :]
        mask = mask[:, input_ids.size(-1)-1:]

        selection_value = torch.gather(logits[:, :-1, :], -1,
                                       model_inputs['input_ids'][:, input_ids.size(-1):, None]).squeeze(-1)

        current_logits = logits[:, :-1, :]
        next_state_value = torch.logsumexp(current_logits.float(), dim=-1)
        next_state_value = next_state_value * mask[:, :-1]
        raw_next_state_value = next_state_value

        scores = selection_value - next_state_value

        assert all((~torch.isinf(scores.view(-1))) & (~torch.isnan(scores.view(-1))))

        assert scores.size() == gen_ids.size()

        return {
            'rewards': scores,
            'inf_mask': inf_mask,
        }
