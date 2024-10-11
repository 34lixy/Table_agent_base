# -*- coding: UTF-8 -*-
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Union

import requests

from LLM_Service.LLM_service import CustomLLM
from tools_set import ToolManager
from tools_set.Rag_tool import RAGService
from until.table_data_preprocess import preprocess_table, get_all_file_paths

os.makedirs('log', exist_ok=True)
log_file_path = os.path.join('log', 'agent_executor.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path, mode='w', encoding='utf-8')])


# logging.StreamHandler()
class AgentExecutor:
    def __init__(self):
        """初始化模型和工具管理器。"""
        self.llm = CustomLLM()
        self.llm_url = "http://10.2.98.108:1220/llm"

        self.tool_manager = ToolManager()
        self.action_des = self.tool_manager.get_tools()
        self.tools_map = self.tool_manager.get_tool_map()

        self.prompt_template = self._load_prompt_file('Prompt/table_system_prompt.txt')
        self.user_prompt = self._load_prompt_file('Prompt/human_prompt.txt')

        self.agent_scratch = ""

    @staticmethod
    def _load_prompt_file(prompt_file_path: str) -> str:
        """加载提示模板文件内容。"""
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"未找到提示模板文件: {prompt_file_path}")
            return ""

    def call_model(self, query: str) -> Optional[Dict[str, Any]]:
        """
        使用给定的查询调用 LLM，并返回响应。
        返回:
            Optional[Dict[str, Any]]: 模型响应字典，如果出错则返回 None。
        """
        try:
            response = self.llm.chat(query, self.user_prompt)
            return response if isinstance(response, dict) else json.loads(response)
        except json.JSONDecodeError as e:
            logging.error(f"解析模型响应出错: {e}")
        except Exception as e:
            logging.error(f"模型调用或其他异常: {e}")
        return None

    def gen_port(self, query: str) -> Optional[Dict[str, Any]]:
        """
        向 LLM 发送系统和用户提示，并返回结果。
        """
        payload = {"sys_prompt": query, 'user_input': self.user_prompt}
        try:
            response = requests.post(self.llm_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"请求错误: {e}")
        return None

    def execute_action(self, tool_name: str, tool_args: Dict[str, Any]) -> Union[Any, str]:
        """
        使用工具管理器执行指定的动作。
        参数:
            tool_name (str): 工具名称。
            tool_args (Dict[str, Any]): 工具参数字典。
        返回:
            Union[Any, str]: 工具函数的返回值或错误信息。
        """
        func = self.tools_map.get(tool_name)
        if not func:
            logging.error(f"未找到对应的工具函数: {tool_name}")
            return f"未找到对应的工具函数: {tool_name}"

        try:
            return func(**tool_args)
        except Exception as e:
            logging.error(f"执行工具函数时出错: {e}")
            return str(e)

    def init_agent_scratch(self) -> None:
        """初始化 Agent 的思考过程。"""
        self.agent_scratch = ""

    def get_agent_scratch(self) -> Optional[str]:
        """ 获取 Agent 思考过程 """
        return self.agent_scratch

    def agent_execute(self, query: str, table_des: str = '', max_request_time: int = 10) -> Optional[str]:
        """
        Agent 主要执行循环。
        参数:
            query (str): 用户问题。
            table_des (str, optional): 表格描述。默认为空字符串。
            max_request_time (int): 最大请求次数。默认为 10。
        返回:
            Optional[str]: 最终答案字符串，如果任务失败则返回 None。
        """
        self.init_agent_scratch()
        prompt = self._format_prompt(query, table_des)

        logging.info(f"系统提示:\n{prompt}")
        start_time = time.time()

        for attempt in range(max_request_time):
            logging.info(f"第 {attempt + 1} 轮: 开始调用模型")

            cot_prompt = prompt.replace('[agent_scratch]', self.agent_scratch)
            response = self._get_model_response(cot_prompt)

            if not response:
                logging.warning("模型响应为空，继续下一轮...")
                continue

            # 处理模型响应
            final_answer = self._handle_response(response)
            if final_answer:
                elapsed_time = time.time() - start_time
                logging.info(f"最终答案: {final_answer}\n总耗时: {elapsed_time:.2f}s")
                return final_answer

        logging.error(f"任务执行失败! 总耗时: {time.time() - start_time:.2f}s。")
        return None

    def _format_prompt(self, query: str, table_des: str) -> str:
        """格式化提示模板。"""
        return self.prompt_template.format(Tools=self.action_des, question=query, DATA_DESC=table_des)

    def _get_model_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """调用模型并返回响应。"""
        start_time = time.time()
        response = self.call_model(prompt)
        logging.info(f"调用模型耗时: {time.time() - start_time:.2f}s")
        return response

    def _handle_response(self, response: Dict[str, Any]) -> Optional[str]:
        """处理模型响应，判断是否为最终答案，并记录思考过程。"""
        thoughts = response.get("thoughts", "")
        action_info = response.get("action", {})
        tool_name = action_info.get("name", "")
        tool_args = action_info.get("args", {})

        if tool_name == "Final Answer":
            final_answer = tool_args.get("answer", "")
            self.agent_scratch += f"\n思考: {thoughts}\n最终结果: {final_answer}\n"
            return final_answer

        # 执行工具函数并更新思考过程
        call_result = self.execute_action(tool_name, tool_args)
        agent_scratch = f"\n思考: {thoughts}\n行动: {action_info}\n观察: {call_result}\n"
        self.agent_scratch += agent_scratch
        print(agent_scratch)
        logging.info(agent_scratch)
        return None


if __name__ == '__main__':
    file_path = 'data'
    des = preprocess_table(file_path)

    file_list = get_all_file_paths(file_path)
    rag = RAGService()
    rag.initialize_vector_store(file_list)

    request_time = 10
    agent = AgentExecutor()
    # query = "帮我写一份2024年3月36大中城市居民消费价格分析数据报告"

    while True:
        user_input = input("请输入您的目标:")
        if user_input == "exit":
            break
        output = agent.agent_execute(user_input, table_des=des)
        chat_cot = agent.get_agent_scratch()
        print('最终答案：', output)
