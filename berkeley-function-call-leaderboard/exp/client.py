import httpx
import openai
import rich


def main():
    client = openai.OpenAI(
        api_key="wqCr1d2Rr86MM5TldlmwPfh8txgn6FSl66K9RGq84Rl8gssD4g7w61Z5AD6N7t2zrkZrZnpdFh0m7j968tcP2z6M64p48Z9BfPxT84jrFSnd648A7688jGq8cxkKl868",
        base_url="https://modelfactory.lenovo.com/service-large-463-1767592802178/llm/v1",
        http_client=httpx.Client(verify=False)
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "unit"]
            }
        }
    }]
    # stream_options = {"include_usage": True}
    result = client.chat.completions.create(
        model="Qwen3-8B-looptool-v0",
        messages=[{"role": "user", "content": "北京的和上海的天气怎么样"}],
        tools=tools,
        stream=False,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False, "thinking": True},
        },
        temperature=0.3,
        max_completion_tokens=2048
    )
    rich.print(result)


if __name__ == "__main__":
    main()