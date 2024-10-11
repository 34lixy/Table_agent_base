import socket

import requests

# 获取本机名称
hostname = socket.gethostname()

# 获取本机IP地址
ip_address = socket.gethostbyname(hostname)

print(f"本机名称：{hostname}")
print(f"本机IP地址：{ip_address}")


def gen_port(text):
    tex = {
        "input": text
    }
    url = "http://10.2.98.108:1220/llm"

    try:
        res = requests.post(url, json=tex)
        res.raise_for_status()  # 抛出HTTPError异常，如果请求返回不是200 OK

        result = res.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None


x = gen_port("你好")

if x is not None:
    print(x)
else:
    print("请求失败")
