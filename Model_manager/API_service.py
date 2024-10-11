import json
import os
import httpx
from openai import OpenAI


# 配置 OpenAI 服务
API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL')
model_name = 'qwen2.5-32b-instruct'


class CustomLLM:
    def __init__(self,
                 model: str = model_name,
                 api_key: str = API_KEY,
                 base_url: str = BASE_URL):

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(verify=False)
        )

        self.max_retry_time = 3

    def chat(self, sys_prompt='', user_input=''):
        cur_retry_time = 0
        response_content = {}
        while cur_retry_time < self.max_retry_time:
            cur_retry_time += 1
            try:
                messages = [{"role": "system", "content": sys_prompt},
                            {'role': 'user', 'content': user_input}]

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0.2,
                    messages=messages
                )
                print(completion)
                result = json.loads(completion.model_dump_json())
                response_content = result['choices'][0]['message']['content']
                return response_content

            except Exception as e:
                print(e)

        return response_content



