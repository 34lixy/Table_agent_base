# -*- coding: utf-8 -*-
import copy
import sys
import json
from openai import OpenAI
import os
import httpx
from Model_manager.API_service import CustomLLM

llm_client = CustomLLM()


def summarize_content_prompt(content, user_name, boot_name, language='cn'):
    """根据对话内容生成总结提示"""
    header = '请总结以下的对话内容，尽可能精炼，提取对话的主题和关键信息。如果有多个关键事件，可以分点总结。对话内容：'

    summary_prefix = '总结' if language == 'cn' else 'Summarization'
    prompt = header if language == 'cn' else header
    for dialog in content:
        query = dialog['query'].strip()
        response = dialog['response'].strip()
        prompt += f"\n{user_name}：{query}\n{boot_name}：{response}"
    prompt += f"\n{summary_prefix}：" if language == 'cn' else f"\n{summary_prefix}:"
    return prompt


def summarize_memory(memory_file_path,
                     name=None,
                     language='cn'):
    boot_name = 'AI'
    gen_prompt_num = 1
    memory = json.loads(open(memory_file_path, 'r', encoding='utf8').read())

    all_prompts, all_his_prompts, all_person_prompts = [], [], []
    for user_name, v in memory.items():
        if name is not None and user_name != name:
            continue

        print(f'Updating memory for user {user_name}')

        if v.get('history') is None:
            continue

        history = v['history']

        if v.get('summary') is None:
            memory[user_name]['summary'] = {}

        if v.get('personality') is None:
            memory[user_name]['personality'] = {}

        for date, content in history.items():
            print(f'Updating memory for date {date}')

            his_flag = False if (date in v['summary'].keys() and v['summary'][date]) else True
            person_flag = False if (date in v['personality'].keys() and v['personality'][date]) else True

            his_prompt = summarize_content_prompt(content, user_name, boot_name, language)
            person_prompt = summarize_person_prompt(content, user_name, boot_name, language)

            if his_flag:
                his_summary = llm_client.generate_text_simple(prompt=his_prompt,
                                                              prompt_num=gen_prompt_num,
                                                              language=language)
                memory[user_name]['summary'][date] = {'content': his_summary}

            if person_flag:
                person_summary = llm_client.generate_text_simple(prompt=person_prompt, prompt_num=gen_prompt_num,
                                                                 language=language)
                memory[user_name]['personality'][date] = person_summary

        overall_his_prompt = summarize_overall_prompt(list(memory[user_name]['summary'].items()), language=language)
        overall_person_prompt = summarize_overall_personality(list(memory[user_name]['personality'].items()),
                                                              language=language)
        memory[user_name]['overall_history'] = llm_client.generate_text_simple(prompt=overall_his_prompt,
                                                                               prompt_num=gen_prompt_num,
                                                                               language=language)
        memory[user_name]['overall_personality'] = llm_client.generate_text_simple(prompt=overall_person_prompt,
                                                                                   prompt_num=gen_prompt_num,
                                                                                   language=language)

    with open(memory_file_path, 'w', encoding='utf8') as file:
        json.dump(memory, file, ensure_ascii=False, indent=4)
        print(f'Successfully update memory for {name}')

    return memory


if __name__ == '__main__':
    memory_file_path = '../eval_data/cn/memory_bank_cn.json'

    result_memory = summarize_memory(memory_file_path, language='cn')

    with open(memory_file_path, 'r', encoding='utf8') as f:
        updated_memory = json.load(f)

    print("Updated memory data:", json.dumps(updated_memory, indent=4, ensure_ascii=False))
