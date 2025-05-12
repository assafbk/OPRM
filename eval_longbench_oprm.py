import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader


def collate_fn(data):
    batch = {}
    batch['query'] = data[0]['input']
    batch['context'] = data[0]['context']
    batch['answers'] = data[0]['answers']
    batch['all_classes'] = data[0]['all_classes']
    batch['length'] = data[0]['length']
    
    return batch

def get_oprm_config(inference_mode, dataset_name, model_type):
    if model_type in ['falcon_mamba', 'falcon3_mamba']:
        pad_token_id = 29104
        idk_token = 5801
    if model_type in ['recurrent_gemma']:
        pad_token_id = 0
        idk_token = 2876
    if model_type in ['rwkv']:
        pad_token_id = 261
        idk_token = [33079, 36638] # 'Error', ' Error'
    
    oprm_config = {
        'inference_mode': inference_mode,
        'ctx_len_toks': -1,
        'num_pre_ctx_tokens': -1,
        'chunk_selection_method': dataset_to_select_method(dataset_name),
        'dataset': 'longbench',
        'parallelize_chunks': inference_mode == 'oprm',
        'chunk_size': 2000,
        'pad_token_id': pad_token_id,
        'model_type': model_type,
        'idk_filter_active': model_type in ['recurrent_gemma', 'falcon_mamba', 'rwkv'],
        'idk_token': idk_token
    }

    return oprm_config

def dataset_to_select_method(dataset_name):
    if dataset_name in ["multi_news", "qmsum", "gov_report"]:
        return 'entropy_first_tok_summ'
    else:
        return 'entropy_first_tok'

def load_model(device, model_type, cache_dir):

    if model_type == 'falcon_mamba':
        model_name = "tiiuae/falcon-mamba-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype, cache_dir=cache_dir)
    elif model_type == 'falcon3_mamba':
        model_name = "tiiuae/Falcon3-Mamba-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b-instruct")
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype, cache_dir=cache_dir)
    elif model_type == 'recurrent_gemma':
        model_name = "google/recurrentgemma-9b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_bos_token = False
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype, cache_dir=cache_dir)
    elif model_type == 'rwkv':
        model_name = "RWKV/v6-Finch-7B-HF"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, torch_dtype=torch.float16).to(device)

    return model, tokenizer

def prepare_input_ids(batch, tokenizer, device, dataset, oprm_config, model_type, context=None):

    if model_type in ['falcon_mamba', 'falcon3_mamba']:
        start_of_context = [8, 9, 5584, 193]    # '<|begin_of_text|><|im_start|>user\n'
        end_of_context = [10, 193, 9, 7, 193]   # '<|im_end|>\n<|im_start|>assistant\n'
    if model_type in ['recurrent_gemma']:
        start_of_context = [2, 106, 1645, 108]    # '<bos><start_of_turn>user\n'
        end_of_context = [107, 108, 106, 2516, 108]   # '<end_of_turn>\n<start_of_turn>model\n'
    if model_type in ['rwkv']:
        start_of_context = [62994, 59, 33]    # 'Instruction: '
        end_of_context = [261, 53671, 59]   # '\n\nResponse:'
    
    if dataset == 'narrativeqa':
        pre_context_query = tokenizer.encode('You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: ')
        if oprm_config['idk_filter_active'] and oprm_config['inference_mode'] == 'oprm':
            query = tokenizer.encode('\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation. If the answer does not exist in the passages, return "Error".\n\nQuestion: ' + batch['query'] + '\n\nAnswer: ')
        else:
            query = tokenizer.encode('\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: ' + batch['query'] + '\n\nAnswer: ')
        
    if dataset in ['hotpotqa', '2wikimqa', 'musique']:
        pre_context_query = tokenizer.encode('Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n ')
        if oprm_config['idk_filter_active'] and oprm_config['inference_mode'] == 'oprm':
            query = tokenizer.encode('\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words. If the answer does not exist in the passages, return "Error".\n\nQuestion: ' + batch['query'] + '\nAnswer: ')
        else:
            query = tokenizer.encode('\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: ' + batch['query'] + '\nAnswer: ')
    
    if dataset == 'qasper': # Note: we tell the model to output "Error" instead of "Unanswerable" (1 tok vs 3 toks). We need to map all Error to Unanswerable when finished so the Qasper score will work.
        if oprm_config['idk_filter_active'] and oprm_config['inference_mode'] == 'oprm':
            pre_context_query = tokenizer.encode('You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "Error". If the question is a yes/no question, answer \"yes\", \"no\", or "Error". Do not provide any explanation.\n\nArticle: ')
            query = tokenizer.encode('\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "Error". If the question is a yes/no question, answer \"yes\", \"no\", or "Error". Do not provide any explanation.\n\nQuestion: ' + batch['query'] + '\n\nAnswer:')
        else:
            pre_context_query = tokenizer.encode('You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: ')
            query = tokenizer.encode('\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: ' + batch['query'] + '\n\nAnswer:')

    if dataset == 'multifieldqa_en':
        pre_context_query = tokenizer.encode('Read the following text and answer briefly.\n\n')
        if oprm_config['idk_filter_active'] and oprm_config['inference_mode'] == 'oprm':
            query = tokenizer.encode('\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words. If the answer does not exist in the passages, return "Error".\n\nQuestion: ' + batch['query'] + '\nAnswer: ')
        else:
            query = tokenizer.encode('\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: ' + batch['query'] + '\nAnswer: ')
    
    if dataset == 'gov_report':
        if model_type == 'recurrent_gemma':
            pre_context_query = tokenizer.encode('You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n')
            query = tokenizer.encode('\n\nNow, write a one-page summary of the report. Only give me the answer and do not output any other words.\n\nSummary: ')
        else:
            pre_context_query = tokenizer.encode('You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n')
            query = tokenizer.encode('\n\nNow, write a one-page summary of the report.\n\nSummary: ')
    
    if dataset == 'qmsum':
        if model_type == 'recurrent_gemma':
            pre_context_query = tokenizer.encode('You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n')
            query = tokenizer.encode('\n\nNow, answer the query based on the above meeting transcript in one or more sentences. Only give me the answer and do not output any other words.\n\nQuery: ' + batch['query'] + '\nAnswer: ')
        else:
            pre_context_query = tokenizer.encode('You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n')
            query = tokenizer.encode('\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: ' + batch['query'] + '\nAnswer: ')
    
    if dataset == 'multi_news':
        if model_type == 'recurrent_gemma':
            pre_context_query = tokenizer.encode('You are given several news passages. Write a one-page summary of all news. \n\nNews:\n')
            query = tokenizer.encode('\n\nNow, write a one-page summary of all the news. Only give me the answer and do not output any other words.\n\nSummary: ')
        else:
            pre_context_query = tokenizer.encode('You are given several news passages. Write a one-page summary of all news. \n\nNews:\n')
            query = tokenizer.encode('\n\nNow, write a one-page summary of all the news.\n\nSummary: ')

    if dataset == 'trec':
        pre_context_query = tokenizer.encode('Please determine the type of the question below. Here are some examples of questions.\n\n')
        query = tokenizer.encode('\n' + batch['query'])
    
    if dataset == 'triviaqa':
        pre_context_query = tokenizer.encode('Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n')
        query = tokenizer.encode('\n\n' + batch['query'])

    if dataset == 'samsum':
        pre_context_query = tokenizer.encode('Summarize the dialogue into a few short sentences. The following are some examples.\n\n')
        query = tokenizer.encode('\n\n' + batch['query'])
    
    if dataset == 'passage_count':
        pre_context_query = tokenizer.encode('There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n')
        query = tokenizer.encode('\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ')
    
    if dataset == 'passage_retrieval_en':
        if oprm_config['idk_filter_active'] and oprm_config['inference_mode'] == 'oprm':
            pre_context_query = tokenizer.encode('Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n')
            query = tokenizer.encode('\n\nThe following is an abstract.\n\n' + batch['query'] + '\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc. If the abstract is not from any of the paragraphs, write "Error".\n\nThe answer is: ')
        else:
            pre_context_query = tokenizer.encode('Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n')
            query = tokenizer.encode('\n\nThe following is an abstract.\n\n' + batch['query'] + '\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ')
    
    if dataset == 'lcc':
        pre_context_query = tokenizer.encode('Please complete the code given below. \n')
        if model_type == 'falcon3_mamba':
            query = tokenizer.encode('Now complete the next line of code. Only give me the answer and do not output any other words:\n')
        else:
            query = tokenizer.encode('Next line of code:\n')
    
    if dataset == 'repobench-p':
        pre_context_query = tokenizer.encode('Please complete the code given below. \n')
        if model_type == 'falcon3_mamba':
            query = tokenizer.encode(batch['query'] + 'Now complete the next line of code. Only give me the answer and do not output any other words:\n')
        else:
            query = tokenizer.encode(batch['query'] + 'Next line of code:\n')

    if model_type == 'rwkv':
        pre_context_query = pre_context_query + [261, 33192, 59, 33]  # \n\nInput: 

    if context is None:
        context = tokenizer.encode(batch['context'])
    
    if dataset == 'lcc' and oprm_config['inference_mode'] == 'oprm':
        context = tokenizer.encode(batch['context'])
        context = context[-oprm_config['chunk_size']:]

    oprm_config['num_pre_ctx_tokens'] = len(start_of_context) + len(pre_context_query)
    oprm_config['ctx_len_toks'] = len(start_of_context) + len(pre_context_query) + len(context)
    prompt = [start_of_context + pre_context_query + context + query + end_of_context]
    input_ids = torch.tensor(prompt).to(device)

    return input_ids


_datasets = ["hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", "multifieldqa_en", \
             "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", \
             "passage_retrieval_en", "lcc", "repobench-p"]

_datasets_e = ["hotpotqa", "2wikimqa", "qasper", "multifieldqa_en", "gov_report", "multi_news", "trec", \
               "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_oprm", type=int, default=-1)
    parser.add_argument("--device", type=str, default='None')
    parser.add_argument("--model", type=str, default='None')
    parser.add_argument("--e", type=int, default=0)
    args = parser.parse_args()

    device=args.device
    is_oprm = args.is_oprm == 1
    model_type = args.model
    assert model_type in ['falcon_mamba', 'falcon3_mamba', 'recurrent_gemma', 'rwkv']
    inference_mode = 'oprm' if is_oprm else 'vanilla'
    e = args.e == 1

    
    # Additional Configurations
    cache_dir = './hf_cache'
    out_path_base = f'./results/LongBench/{model_type}/{inference_mode}'
    Path(out_path_base).mkdir(parents=True, exist_ok=True)

    max_len_per_seg = 120000 # OPRM: longest sequence length we allow on the gpu (allows parallelization of very long sequences by processing max_len_per_seg/chunk_sizes_to_test chunks at a time)
    if is_oprm:
        chunk_sizes_to_test = [1000, 2000, 3000]
        assert(all([max_len_per_seg % chunk_size_to_test == 0 for chunk_size_to_test in chunk_sizes_to_test])) # guarantees that no chunks will be sliced
    else:
        chunk_sizes_to_test = [-1] # just for generality

    datasets_to_test = ["hotpotqa",  "2wikimqa", "qasper", "multifieldqa_en"]
    
    dataset_ntoks = {"hotpotqa": 20, "2wikimqa": 20, "musique": 20, "narrativeqa": 20, "qasper": 20, "multifieldqa_en": 20,   \
                     "gov_report": 500, "qmsum": 250, "multi_news": 250, "trec": 20, "triviaqa": 20, \
                     "samsum": 250, "passage_count": 20, "passage_retrieval_en": 20, "lcc": 40, "repobench-p": 40}

    
    # Run LongBench
    model, tokenizer = load_model(device, model_type, cache_dir)
    for dataset_name in datasets_to_test:
        
        if e and dataset_name not in _datasets_e:
            continue

        oprm_config = get_oprm_config(inference_mode, dataset_name, model_type)

        for chunk_size in chunk_sizes_to_test:
        
            start_datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            print(f'\n\nCurrent Dataset: {dataset_name}, Model Type: {model_type}, Inference Mode: {inference_mode}, Chunk Size: {chunk_size}, is LB_e: {e}\n\n')

            oprm_config['chunk_size'] = chunk_size
            max_gen_len = dataset_ntoks[dataset_name]

            if e:
                data = load_dataset('THUDM/LongBench', f'{dataset_name}_e', cache_dir=cache_dir)
            else:
                data = load_dataset('THUDM/LongBench', f'{dataset_name}', cache_dir=cache_dir)

            if inference_mode == 'oprm':
                if e:
                    out_path = f'{out_path_base}/{dataset_name}.{start_datetime_str}_{inference_mode}_{max_gen_len}_toks_cs_{oprm_config["chunk_size"]}_type_e.jsonl'
                else:
                    out_path = f'{out_path_base}/{dataset_name}.{start_datetime_str}_{inference_mode}_{max_gen_len}_toks_cs_{oprm_config["chunk_size"]}.jsonl'
            else:
                if e:
                    out_path = f'{out_path_base}/{dataset_name}.{start_datetime_str}_{inference_mode}_{max_gen_len}_toks_type_e.jsonl'
                else:
                    out_path = f'{out_path_base}/{dataset_name}.{start_datetime_str}_{inference_mode}_{max_gen_len}_toks.jsonl'

            dataloader = DataLoader(data['test'], collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=0)
            for idx, batch in enumerate(tqdm(dataloader)):           
                if is_oprm:
                    out = []
                    context = tokenizer.encode(batch['context'])
                    remaining = min(len(context), max_len_per_seg)
                    cur_min_ent = torch.inf
                    while len(context) > 0:
                        cur_input_ids = prepare_input_ids(batch, tokenizer, device, dataset_name, oprm_config, model_type, context[:remaining])
                        cur_out = model.generate(
                            input_ids=cur_input_ids,
                            max_new_tokens=max_gen_len,
                            eos_token_id=tokenizer.eos_token_id,
                            oprm_config=oprm_config
                        )
                        if oprm_config['chunk_selection_method'] == 'entropy_first_tok_summ':
                            out.append(cur_out.sequences.clone())
                        
                        elif oprm_config['chunk_selection_method'] == 'entropy_first_tok' and cur_min_ent > oprm_config['min_ent']:
                            cur_min_ent = oprm_config['min_ent']
                            out = cur_out.clone()
                        
                        context = context[remaining:]
                        remaining = min(len(context), max_len_per_seg)
                else:
                    input_ids = prepare_input_ids(batch, tokenizer, device, dataset_name, oprm_config, model_type)
                    out = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_gen_len,
                        eos_token_id=tokenizer.eos_token_id,
                        oprm_config=oprm_config
                    )
                
                if oprm_config['chunk_selection_method'] == 'entropy_first_tok_summ':
                    out = torch.vstack(out)
                    decoded = tokenizer.batch_decode(out)
                    pred = ''
                    for i in range(len(decoded)):
                        if model_type in ['falcon_mamba', 'falcon3_mamba']:
                            pred += decoded[i].split("<|im_start|>assistant\n")[1].split(f'<|im_end|>\n{tokenizer.eos_token}')[0]
                        if model_type == 'recurrent_gemma':
                            pred += decoded[i].split("<end_of_turn>\n<start_of_turn>model\n")[1].split(f'{tokenizer.eos_token}')[0]
                        if model_type == 'rwkv':
                            pred += decoded[i].split("\n\nResponse:")[1].split(f'{tokenizer.eos_token}')[0]

                        pred += '\n'
                else:
                    decoded = tokenizer.batch_decode(out)
                    if model_type in ['falcon_mamba', 'falcon3_mamba']:
                        pred = (decoded[0].split("<|im_start|>assistant\n")[1].split(f'<|im_end|>\n{tokenizer.eos_token}')[0])
                    if model_type in ['recurrent_gemma']:
                        pred = (decoded[0].split("<end_of_turn>\n<start_of_turn>model\n")[1].split(f'{tokenizer.eos_token}')[0])
                    if model_type == 'rwkv':
                        pred = (decoded[0].split("\n\nResponse:")[1].split(f'{tokenizer.eos_token}')[0])

                with open(out_path, "a", encoding="utf-8") as f:
                    json.dump({"pred": pred, "answers": batch['answers'], "all_classes": batch['all_classes'], "length": batch["length"]}, f, ensure_ascii=False)
                    f.write('\n')