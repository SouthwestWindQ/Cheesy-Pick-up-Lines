import torch
import jieba
from config import args
from transformers import XLNetTokenizer, AutoModelForCausalLM


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'mymusise/CPM-Generate-distill'
tokenizer = XLNetTokenizer.from_pretrained(model_name)
cls, sep = tokenizer.cls_token, tokenizer.sep_token
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.resize_token_embeddings(len(tokenizer))
checkpoint = torch.load('model/finalmodel.pth', map_location=torch.device(device))
model.load_state_dict(checkpoint['net'])

for i in range(10):
    keyword = cls + input('关键词{}:'.format(i+1)) + sep
    input_ids = tokenizer(keyword)['input_ids'][:-2]
    input_ids = torch.tensor(input_ids).resize(1,len(input_ids)).to(device)
    count = 0
    while True:
        output_temp = model.generate(input_ids, max_length=args.max_length, do_sample=True, 
                                     temperature=args.temperature, top_k=0)
        generated = tokenizer.decode(output_temp[0])[len(keyword):]
        start = 0
        for i, c in enumerate(generated):
            if not c in (',', '，', ' ', '\n', ':', '：', '<', 's', 'e', 'p', 'c', 'l', '>'):
                start = i
                break
        generated = generated[start:]         
        if generated == "":
            continue
        count += 1
        print('土味情话{}:'.format(count), end=' ')
        for word in generated:
            print(word, end='')
            if word == '。' or word == '！' or word == '!':
                break
        print()
        if count == 5:
            break

