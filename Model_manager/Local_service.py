import socket
import requests


class LocalLLM:
    def __init__(self, llm_url: str = "http://10.2.98.108:1220/llm"):
        self.url = llm_url

    def chat(self,
             sys_prompt: str = '',
             user_prompt: str = ''):

        payload = {"sys_prompt": sys_prompt,
                   'user_input': user_prompt}

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.url,
                                     json=payload,
                                     headers=headers)
            ans = response.json()
            return ans['result']
        except requests.RequestException as e:
            print(f"请求错误: {e}")
        return None


if __name__ == '__main__':
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    print(f"本机名称：{hostname}")
    print(f"本机IP地址：{ip_address}")
    lm = LocalLLM()
    print(lm.chat("You are a helpful assistant.","Tell me a joke."))

