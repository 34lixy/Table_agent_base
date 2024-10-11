import json
import os

import httpx
from flask import Flask, request, Response
from gevent import monkey, pywsgi
from openai import OpenAI

# monkey.patch_all(thread=False)

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
                result = json.loads(completion.model_dump_json())
                response_content = result['choices'][0]['message']['content']
                return response_content

            except Exception as e:
                response_content = {}

            return response_content


def start_server(http_id, port):
    gen = CustomLLM()
    print('服务已启动')
    app = Flask(__name__)

    @app.route("/")
    def index():
        return "Getting Started"

    @app.route("/llm", methods=["GET", "POST"])
    def generate():
        response = {
            "success": False,
        }

        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                print(f"arg_dict:{arg_dict}")
                if "input" in arg_dict and isinstance(arg_dict["input"], str):
                    sys_prompt = arg_dict["sys_prompt"]
                    user_input = arg_dict["user_input"]
                    result = gen.chat(sys_prompt, user_input)
                    print(f"result:{result}")
                    response = {
                        "success": True,
                        "result": result,
                        "input": sys_prompt
                    }
                else:
                    response = {
                        "success": False,
                        "error": "Invalid input format"
                    }

        except Exception as e:
            response = {
                "success": False,
                "error": str(e)
            }

        return Response(json.dumps(response, ensure_ascii=False), content_type="application/json")

    sever = pywsgi.WSGIServer((str(http_id), port), app)
    sever.serve_forever()


if __name__ == '__main__':
    start_server("0.0.0.0", 1220)
