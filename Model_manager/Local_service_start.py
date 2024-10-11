from flask import Flask, request, Response
from gevent import monkey, pywsgi
import logging
from API_service import CustomLLM
import json

monkey.patch_all(thread=False)


def start_server(http_id, port):
    logging.basicConfig(level=logging.INFO)

    llm = CustomLLM()  # 假设 CustomLLM 已定义
    logging.info('服务已启动')
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
            if request.content_type == "application/json":
                arg_dict = request.get_json()
                logging.info(f"Received data: {arg_dict}")

                sys_prompt = arg_dict.get("sys_prompt")
                user_input = arg_dict.get("user_input")

                result = llm.chat(sys_prompt, user_input)

                response = {
                    "success": True,
                    "result": result,
                    "sys_prompt": sys_prompt,
                    "user_input": user_input
                }
                logging.info(f"Response: {response}")
            else:
                response = {
                    "success": False,
                    "error": "Invalid input format"
                }

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            response = {
                "success": False,
                "error": str(e)
            }

        return Response(json.dumps(response, ensure_ascii=False), content_type="application/json")

    server = pywsgi.WSGIServer((http_id, port), app)
    server.serve_forever()


if __name__ == '__main__':
    start_server("0.0.0.0", 1220)
