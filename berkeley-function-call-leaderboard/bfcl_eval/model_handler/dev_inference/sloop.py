import json
import re
from typing import Any
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override

# 系统提示词模板
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
        # 151645: <|im_end|>
        # 151643: <|endoftext|>
        self.stop_token_ids = [151645, 151643] 

    @override
    def _format_prompt(self, messages, function):
        """
        将 BFCL 的 messages 列表转换为 Sloop 微调格式 (Qwen ChatML) 的 Prompt 字符串。
        关键特性：
        1. 注入工具定义。
        2. 处理思维链 (Thinking Process)。
        3. 聚合 Tool Response 为 <tool_response>[...]</tool_response>。
        """
        formatted_prompt = ""

        # 1. 注入 System Prompt
        tools_json = json.dumps(function, ensure_ascii=False)
        system_content = SYSTEM_PROMPT_TEMPLATE.format(tool_definitions=tools_json)
        formatted_prompt += f"<|im_start|>system\n{system_content}<|im_end|>\n"

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # 跳过，前面已经手动注入了
                i += 1
                continue

            elif role == "user":
                formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                i += 1

            elif role == "assistant":
                # 处理思维链
                reasoning_content = msg.get("reasoning_content", "")
                
                # 如果 content 为空但有 tool_calls，我们需要把 tool_calls 还原回模型输出格式
                # BFCL 可能会把 assistant 的回复拆成 content 和 tool_calls 两个字段
                # 你的模型输出格式是: [func1(), func2()]
                
                final_content = content
                if "tool_calls" in msg and msg["tool_calls"]:
                    # 这里我们需要把 BFCL 解析好的 tool_calls 逆向还原成字符串列表
                    # msg['tool_calls'] 通常是 list[dict] 或者 list[str] (取决于之前的处理)
                    # OSSHandler 的流程中，这里的 tool_calls 可能是被 decode_execute 处理过的 List[str]
                    # 例如: ["releaseBrakePedal()", "gallon_to_liter(gallon=21.0)"]
                    
                    tool_calls_list = msg["tool_calls"]
                    # 如果 tool_calls 是字典列表（Native FC 格式），需要转换（虽然你的模型是 Prompting 模式，不太可能走到这步，但防一手）
                    if tool_calls_list and isinstance(tool_calls_list[0], dict):
                         # 这里是个大坑，因为你的格式是 String，不是 JSON
                         # 暂时假设在 Prompting 模式下，content 已经包含了 "[func(), func()]"
                         # 如果 content 是空的，说明是 history 回填，我们需要重构它
                         pass
                
                # 如果 content 本身就是空的（只有 tool_calls），我们需要依赖 tool_calls 重建 content
                # 但在 Prompting 模式下，BFCL 通常会把完整的模型输出存入 content
                # 如果 content 不为空，直接用 content
                
                if reasoning_content:
                    formatted_prompt += f"<|im_start|>assistant\n<think>\n{reasoning_content}\n</think>\n\n{final_content}<|im_end|>\n"
                else:
                    # 如果内容里已经包含了 </think>，就不再额外包裹
                    if "</think>" in final_content:
                         formatted_prompt += f"<|im_start|>assistant\n{final_content}<|im_end|>\n"
                    else:
                         formatted_prompt += f"<|im_start|>assistant\n{final_content}<|im_end|>\n"
                i += 1

            elif role == "tool":
                # --- 核心逻辑：聚合连续的 Tool 消息 ---
                tool_results_list = []
                
                # 循环读取所有连续的 tool 消息
                while i < len(messages) and messages[i]["role"] == "tool":
                    curr_msg = messages[i]
                    # name: "releaseBrakePedal()"
                    # content: '{"brakePedalStatus": "released", ...}'
                    func_call_sign = curr_msg.get("name", "unknown_tool()") 
                    exec_result = curr_msg.get("content", "")

                    # 构造字典项: { "func_call()": "result_json_str" }
                    tool_results_list.append({func_call_sign: exec_result})
                    i += 1
                
                # 序列化为 User 回复
                # 使用 str() 生成单引号格式: [{'k': 'v'}]，匹配微调数据
                tool_response_str = str(tool_results_list)
                
                formatted_prompt += f"<|im_start|>user\n<tool_response>\n{tool_response_str}\n</tool_response><|im_end|>\n"
                
            else:
                # 兜底
                i += 1

        # 2. 引导 Assistant 生成
        formatted_prompt += "<|im_start|>assistant\n"
        
        return formatted_prompt

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """
        解析模型输出。
        输入 result 示例: 
        1. "[releaseBrakePedal(), gallon_to_liter(gallon=21.0)]<|im_end|>"
        2. "<think>...</think>\n\n[func1()]"
        """
        # 1. 预处理：移除思维链和 Stop Token
        raw_result = result
        if "</think>" in raw_result:
            raw_result = raw_result.split("</think>")[-1].strip()
        
        raw_result = raw_result.strip()
        for stop_token in ["<|im_end|>", "<|endoftext|>"]:
            if raw_result.endswith(stop_token):
                raw_result = raw_result[:-len(stop_token)].strip()

        # 2. 定位最外层的 []
        start_idx = raw_result.find("[")
        end_idx = raw_result.rfind("]")

        if start_idx == -1 or end_idx == -1:
            # 如果没找到 []，可能模型在瞎聊，或者格式错了
            # 对于 BFCL，返回空列表表示没有工具调用
            return []

        # 提取括号内的内容: "func1(), func2()"
        content = raw_result[start_idx+1 : end_idx].strip()
        
        if not content:
            return []

        # 3. 状态机分割 (State Machine Split)
        # 必须处理参数中包含逗号的情况，例如: func(a=[1, 2])
        calls = []
        current_call = ""
        bracket_depth = 0 # 圆括号深度
        square_bracket_depth = 0 # 方括号深度 (处理列表参数)
        curly_bracket_depth = 0 # 花括号深度 (处理字典参数)
        
        # 遍历字符
        for char in content:
            if char == ',' and bracket_depth == 0 and square_bracket_depth == 0 and curly_bracket_depth == 0:
                # 只有在所有括号都在顶层时，逗号才是函数间的分隔符
                if current_call.strip():
                    calls.append(current_call.strip())
                current_call = ""
            else:
                current_call += char
                # 更新深度
                if char == '(': bracket_depth += 1
                elif char == ')': bracket_depth -= 1
                elif char == '[': square_bracket_depth += 1
                elif char == ']': square_bracket_depth -= 1
                elif char == '{': curly_bracket_depth += 1
                elif char == '}': curly_bracket_depth -= 1
        
        # 添加最后一个函数
        if current_call.strip():
            calls.append(current_call.strip())
            
        return calls

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """
        用于 AST 评测。
        输入: "[func(a=1)]"
        输出: [{"func": {"a": 1}}]
        """
        # 复用 decode_execute 拿到字符串列表 ["func(a=1)"]
        decoded_strings = self.decode_execute(result, has_tool_call_tag)
        
        ast_list = []
        for func_str in decoded_strings:
            # convert_to_function_call 能把 "func(a=1)" 解析为 [{"name": "func", "args": {"a": 1}}]
            # 注意它返回的是一个 list，通常包含一个 dict
            parsed = convert_to_function_call(func_str)
            if parsed:
                # BFCL AST Checker 期望的是 [{"func_name": {"arg": val}}] 这种扁平结构
                # 但 convert_to_function_call 返回的是标准结构 {"name":..., "args":...}
                # 让我们检查一下 BFCL 默认的 default_decode_ast_prompting 实现...
                # 实际上 OSSHandler 默认调用的 default_decode_ast_prompting 内部也是用了 convert_to_function_call
                # 所以这里直接 extend 是安全的，只要格式对齐。
                
                # 修正：convert_to_function_call 返回的是 List[Dict]
                # 每个 Dict 是 {"name": "func", "args": {..}}
                # 这正是 BFCL AST 需要的通用格式
                ast_list.extend(parsed)
        
        return ast_list

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        解析 API 响应，提取思维链和最终回复。
        """
        # OSSHandler 使用的是 completions 接口，choices[0].text
        model_response = api_response.choices[0].text
        
        reasoning_content = ""
        cleaned_response = model_response
        
        # 分离 <think> 部分
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            # 提取前半部分，去掉 <think> 标签
            if "<think>" in parts[0]:
                reasoning_content = parts[0].split("<think>")[-1].strip()
            else:
                reasoning_content = parts[0].strip()
            
            # 后半部分是正文
            cleaned_response = parts[-1].strip()
        
        # 构造用于 History 回填的消息对象
        model_responses_message_for_chat_history = {
            "role": "assistant",
            "content": cleaned_response,
            "reasoning_content": reasoning_content
        }

        return {
            "model_responses": model_response, # 传给 decode_execute 的是原始完整字符串
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }