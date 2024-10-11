import inspect

from .Rag_tool import retriever_tool
from .add_sum import add
from .exponential import exponential
from .multiply import multiply
from .split_query import split_query


class ToolManager:
    def __init__(self):
        self.ALL_TOOLS = [
            retriever_tool, split_query, multiply, add, exponential
        ]

    def get_tool_map(self):
        """动态生成工具映射字典，将函数名映射到相应的函数。"""
        return {tool.__name__: tool for tool in self.ALL_TOOLS}

    def get_tools(self):
        """返回所有工具的名称、描述和参数信息的列表。"""
        tools_des = [self.get_function_info(tool) for tool in self.ALL_TOOLS]
        return '\n'.join(tools_des)

    @staticmethod
    def get_function_info(func):
        """获取函数的名称、描述和参数信息。"""

        signature = inspect.signature(func)

        # Prepare the base info
        func_name = func.__name__
        return_type = str(
            signature.return_annotation) if signature.return_annotation != inspect.Signature.empty else "unknown"
        description = func.__doc__.strip() if func.__doc__ else "No description provided"

        # Prepare the arguments info
        args_info = {}
        for param_name, param in signature.parameters.items():
            args_info[param_name] = {
                'name': param_name.replace('_', ' ').title(),  # Convert to a more readable format
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else "unknown"
            }

        # Format the output
        args_format = ', '.join([f"{name}: {info['type']}" for name, info in args_info.items()])
        formatted_output = f"{func_name}({args_format}) -> {return_type} - {description}, args: {args_info}"

        return formatted_output
