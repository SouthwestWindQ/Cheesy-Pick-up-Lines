import pdb
import jieba
import torch
from config import args
from dataHelper import get_dataset
from torch.nn import CrossEntropyLoss
from transformers import TextGenerationPipeline
from transformers import XLNetTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

pad_id = 0

class XLNetTokenizer(XLNetTokenizer):
    translator = str.maketrans(" \n", "\u2582\u2583")
    def _tokenize(self, text, *args, **kwargs):
        text = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        text = " ".join(text)
        return super()._tokenize(text, *args, **kwargs)
    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text


class Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs[0]
        labels = inputs['input_ids']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
        loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    # model_name = 'IDEA-CCNL/Wenzhong-GPT2-3.5B'
    # model_name = 'liam168/gen-gpt2-medium-chinese'
    model_name = 'mymusise/CPM-Generate-distill'
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    datasets = get_dataset()
    cls, sep = tokenizer.cls_token, tokenizer.sep_token
    datasets['train'] = datasets['train'].map( 
        lambda e: tokenizer(
            [cls + e['keyword'][i] + sep + e['sentence'][i] + sep for i in range(len(e['keyword']))],
            truncation=True, padding='max_length', max_length=args.max_length
        ), batched=True,
    )
    datasets['train'].set_format(type='torch', columns=['input_ids'])
    datasets['test'] = datasets['test'].map( 
        lambda e: tokenizer(
            [cls + e['keyword'][i] + sep + e['sentence'][i] + sep for i in range(len(e['keyword']))],
            truncation=True, padding='max_length', max_length=args.max_length
        ), batched=True,
    )
    datasets['test'].set_format(type='torch', columns=['input_ids'])
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    for p in model.parameters():
        p.requires_grad = True
    
    training_args = TrainingArguments(
        learning_rate=args.lr,
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        num_train_epochs=args.epoch,
        weight_decay=args.weight_decay,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
    )
    trainer = Trainer(
        model=model, 
        args=training_args, 
        eval_dataset=datasets['test'],
        train_dataset=datasets['train'],
    )
    trainer.train()
    
    model = model.cpu()
    text_generater = TextGenerationPipeline(model, tokenizer)
    keyword = "晚安"
    print('关键词:', keyword)
    keyword = tokenizer.cls_token + keyword + tokenizer.sep_token
    text = text_generater(keyword, max_length=80, top_k=1, use_cache=True, prefix='')[0]['generated_text']
    text = text[len(keyword):]
    print('土味情话:', end=' ')
    for s in text:
        print(s, end='')
        if s == tokenizer.sep_token or s == '。':
            break


if __name__ == '__main__':
    main()