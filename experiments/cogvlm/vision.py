import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
# argparse is used to parse the command line arguments
import argparse

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--image', type=str, default='/notebooks/scene.png')
    args_parser.add_argument('--query', type=str, default='Describe this image in details.')

    args = args_parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        #torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()

    image = Image.open(args.image).convert('RGB')
    query = args.query
    history = []

    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.float16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0]))