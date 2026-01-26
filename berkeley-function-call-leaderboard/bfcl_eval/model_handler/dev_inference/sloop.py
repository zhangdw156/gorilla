import json
import ast
import re
from typing import Any
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override

SYSTEM_PROMPT_TEMPLATE = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task. 

Here is a list of functions in JSON format that you can invoke.
{tool_definitions}"""

class SloopQwenHandler(OSSHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        # FC models use its own system prompt, so no need to add any message

        return {"message": [], "function": functions}
    
    @staticmethod
    def _extract_tool_calls(input_string):
        """
        从模型输出中提取函数调用并转换为 BFCL 要求的 list[dict] 格式。
        支持格式：[func1(a=1), func2()]
        """
        if "</think>" in input_string:
            input_string = input_string.split("</think>")[-1]
        
        input_string = input_string.strip()

        # 2. 定位方括号边界
        start_idx = input_string.find("[")
        end_idx = input_string.rfind("]")
        
        if start_idx == -1 or end_idx == -1:
            return []

        list_str = input_string[start_idx : end_idx + 1]

        try:
            tree = ast.parse(list_str, mode='eval')
            
            if not isinstance(tree.body, ast.List):
                return []

            tool_calls = []
            for elt in tree.body.elts:
                if isinstance(elt, ast.Call):
                    if isinstance(elt.func, ast.Name):
                        func_name = elt.func.id
                    elif isinstance(elt.func, ast.Attribute):
                        func_name = ast.unparse(elt.func)
                    else:
                        continue

                    arguments = {}

                    for kw in elt.keywords:
                        arguments[kw.arg] = ast.literal_eval(kw.value)

                    tool_calls.append({
                        "name": func_name,
                        "arguments": arguments
                    })
            
            return tool_calls

        except Exception as e:
            return []

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        model_responses_message_for_chat_history = {
            "role": "assistant",
            "content": cleaned_response,
        }
            
            
        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = ""

        # 1. 系统提示词注入
        tools_json = json.dumps(function, ensure_ascii=False)
        system_content = SYSTEM_PROMPT_TEMPLATE.format(tool_definitions=tools_json)
        formatted_prompt += f"<|im_start|>system\n{system_content}<|im_end|>\n"

        # 2. 找到最后一个真正的 User Query 索引 (非工具返回)
        last_query_index = -1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and isinstance(message["content"], str)
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        # 3. 遍历渲染消息
        skip_idx = -1
        for idx, message in enumerate(messages):
            if idx <= skip_idx:
                continue
                
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]
                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = parts[-1].lstrip("\n")

                formatted_prompt += f"<|im_start|>{role}\n"
                
                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"
                
                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                tool_results_list = []
                temp_ptr = idx
                
                while temp_ptr < len(messages) and messages[temp_ptr]["role"] == "tool":
                    curr_msg = messages[temp_ptr]
                    func_call_name = curr_msg.get("name", "unknown_tool()")
                    exec_result = curr_msg.get("content", "")
                    
                    tool_results_list.append({func_call_name: exec_result})
                    temp_ptr += 1
                
                skip_idx = temp_ptr - 1
                
                tool_response_str = str(tool_results_list)
                
                formatted_prompt += f"<|im_start|>user\n<tool_response>\n{tool_response_str}\n</tool_response><|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"    
        return formatted_prompt
    
    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

if __name__ == "__main__":
    # 初始化 Handler
    # 注意：这里传入 Mock 参数，因为我们主要测试静态方法和重写的方法
    handler = SloopQwenHandler(
        model_name="test-model",
        temperature=0.1,
        registry_name="test-reg",
        is_fc_model=True
    )

    print("=== 开始运行 SloopQwenHandler 单元测试 ===\n")

    # --- 测试 1: _extract_tool_calls 解析功能 ---
    print("测试 1: _extract_tool_calls")
    test_cases_extract = [
        {
            "name": "标准单函数调用",
            "input": "[get_weather(location='Shanghai', unit='celsius')]",
            "expected": [{"name": "get_weather", "arguments": {"location": "Shanghai", "unit": "celsius"}}]
        },
        {
            "name": "带思考过程的多函数调用",
            "input": "<think>\n用户想知道天气和股价。\n</think>\n[get_weather(city='Beijing'), get_stock_price(symbol='AAPL')]",
            "expected": [
                {"name": "get_weather", "arguments": {"city": "Beijing"}},
                {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}
            ]
        },
        {
            "name": "空调用或错误格式",
            "input": "我无法完成这个任务。",
            "expected": []
        }
    ]

    for case in test_cases_extract:
        res = handler._extract_tool_calls(case["input"])
        status = "PASSED" if res == case["expected"] else "FAILED"
        print(f"[{status}] {case['name']}")
        if status == "FAILED":
            print(f"  Expected: {case['expected']}\n  Got: {res}")

    print("\n" + "-"*30 + "\n")

    # --- 测试 2: _format_prompt 模板拼装 ---
    print("测试 2: _format_prompt")
    
    test_functions = [{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}]
    test_messages = [
        {"role": "user", "content": "今天上海天气怎么样？"},
        {"role": "assistant", "content": "[get_weather(location='Shanghai')]", "reasoning_content": "分析：用户询问上海天气。"},
        {"role": "tool", "name": "get_weather", "content": '{"temp": 25, "condition": "Sunny"}'},
    ]

    try:
        formatted = handler._format_prompt(test_messages, test_functions)
        
        # 验证是否包含关键标志位
        assertions = {
            "包含 System Prompt": "<|im_start|>system" in formatted,
            "包含工具定义": "tool_definitions" not in formatted and "get_weather" in formatted,
            "包含思考过程": "<think>\n分析：用户询问上海天气。" in formatted,
            "正确聚合 Tool 响应": "<tool_response>\n[{'get_weather': '{\"temp\": 25, \"condition\": \"Sunny\"}'}]\n</tool_response>" in formatted,
            "以 Assistant 引导结尾": formatted.strip().endswith("<|im_start|>assistant")
        }

        for desc, success in assertions.items():
            print(f"[{'PASSED' if success else 'FAILED'}] {desc}")
        
        # 打印部分结果供人工核对
        # print("\n生成的 Prompt 片段预览:\n", formatted[-300:])

    except Exception as e:
        print(f"[FAILED] _format_prompt 运行出错: {str(e)}")

    print("\n" + "-"*30 + "\n")

    # --- 测试 3: decode_ast 接口 ---
    print("测试 3: decode_ast")
    raw_model_output = "[add(a=1, b=2)]"
    try:
        decoded = handler.decode_ast(raw_model_output, "python", False)
        expected_decoded = [{"add": {"a": 1, "b": 2}}]
        if decoded == expected_decoded:
            print("[PASSED] decode_ast 转换成功")
        else:
            print(f"[FAILED] decode_ast 结果不符: {decoded}")
    except Exception as e:
        print(f"[FAILED] decode_ast 出错: {str(e)}")

    print("\n=== 测试完成 ===")