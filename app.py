# app.py
import os
from dotenv import load_dotenv
import chromadb
import dashscope
from dashscope import Generation, TextEmbedding
from dashscope import Generation

# ============ 加载环境变量 ============
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# ====================================================

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1' # 为API请求提供地址

# ========== 配置项 (请与build_vector_db.py保持一致) ==========
KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_DB_DIR = "db/chroma_db"
COLLECTION_NAME = "linux_commands"


# ====================================================

def get_embeddings(texts):
    """
    使用阿里云 text-embedding-v3 模型批量获取文本嵌入向量。
    """
    if not isinstance(texts, list):
        texts = [texts]
        
    response = TextEmbedding.call(
        model='text-embedding-v3',
        input=texts,
        api_key=DASHSCOPE_API_KEY
    )
    
    if response.status_code == 200:
        # 'dense' 是默认的向量类型
        embeddings = [item['embedding'] for item in response.output['embeddings']]
        return embeddings
    else:
        raise Exception(f"嵌入API调用失败: {response.code}, {response.message}")

def retrieve_context(query, n_results=3):
    """
    根据用户查询，从向量数据库中检索最相关的上下文文本块。
    使用阿里云 text-embedding-v3 进行查询。
    """
    try:
        # 1. 为查询文本生成嵌入
        query_embedding = get_embeddings(query)[0] # get_embeddings 返回列表，取第一个
        
        # 2. 连接到数据库并查询
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        results = collection.query(
            query_embeddings=[query_embedding], # 注意这里是 query_embeddings
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []
        
    except Exception as e:
        print(f"检索时出错: {e}")
        return []


def call_qwen_api(prompt, model='qwen3-max'):
    """
    调用通义千问API生成回答。
    """
    # 设置API Key
    Generation.api_key = DASHSCOPE_API_KEY

    try:
        response = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=model,
            prompt=prompt,
            max_tokens=1024,
            temperature=0.5,  # 降低随机性，让回答更稳定
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"❌ API调用失败: {response.code}, {response.message}"

    except Exception as e:
        return f"❌ 发生错误: {str(e)}"


def get_answer(user_question, history_tuples=None):
    """
    主函数：结合RAG和LLM，给出最终答案。
    支持：
    - 从知识库检索（RAG）
    - 使用对话历史作为上下文（记忆）
    - 混合模式：有知识库就用知识库，没有就用通用能力
    """
    if history_tuples is None:
        history_tuples = []

    # ============ 1. 构造对话历史上下文 ============
    history_context = ""
    if history_tuples:
        # 将历史对话转换为自然语言文本，作为上下文
        history_lines = []
        for user_msg, ai_msg in history_tuples:
            history_lines.append(f"用户：{user_msg}")
            history_lines.append(f"AI：{ai_msg}")
        history_context = "\n\n".join(history_lines)
        history_context = f"【过往对话】\n{history_context}\n\n"

    # ============ 2. 尝试从知识库检索 ============
    contexts = retrieve_context(user_question, n_results=2)

    # ============ 3. 构造最终Prompt ============
    if contexts:
        # 有知识库信息：使用RAG模式（高准确）
        context_str = "\n\n".join(contexts)
        final_prompt = f"""你是一个专业的Linux命令学习助手，专门为用户学习Linux命令提供帮助。以下是根据用户查询从知识库中检索到的相关信息：

【参考信息】
{context_str}

{history_context}  # ← 注入历史上下文！

请根据以上提供的知识库内容，为用户解答问题。要求：
1. 严格基于知识库中的信息回答，不要添加知识库中没有的内容
2. 回答要清晰、简洁、专业，适合Linux学习者理解
3. 如果知识库中包含命令示例，请完整展示并解释每个参数的作用
4. 如果知识库中包含多个相关信息，请整合成一个完整的回答
5. 在回答末尾标注信息来源（如：根据Linux命令手册/根据知识库文档）
6. 如果用户询问的命令有安全风险或需要注意事项，请特别提醒
7. 保持教育性和实用性，帮助用户真正理解命令的使用方法

【当前问题】
{user_question}

现在请基于知识库内容给出专业解答："""
    else:
        # 没有知识库信息：使用通用问答模式
        final_prompt = f"""你是一个专业的Linux命令学习助手，专门为用户学习Linux命令提供帮助。经过检索，知识库中没有找到与用户问题相关的具体信息。
{history_context}  # ← 注入历史上下文！

<system_info>
- 当前是通用知识回答模式
- 请基于你的训练知识回答Linux相关问题
- 重点回答技术准确性，避免猜测不确定的信息
- 如果问题超出Linux范围，请友好引导回Linux主题
- 保持专业、教育性的语气
</system_info>

请根据你的通用知识为用户解答问题。要求：
1. 仅回答与Linux命令、系统管理、Shell脚本相关的技术问题
2. 确保技术细节的准确性，特别是命令语法、参数和使用场景
3. 提供实用的示例代码，并解释关键部分
4. 明确标注这是基于通用知识的回答，而非来自特定知识库
5. 如果对某些细节不确定，请说明并建议用户查阅官方文档
6. 对于复杂命令，分步骤解释使用方法
7. 提醒用户在生产环境中使用命令前要充分测试

【当前问题】
{user_question}

现在请基于你的通用Linux知识给出专业解答，并在开头说明"知识库中未找到相关信息，以下基于通用Linux知识解答："："""

    # ============ 4. 调用大模型 ============
    answer = call_qwen_api(final_prompt)
    return answer


# =================== 运行测试 ===================
try:
    import gradio as gr

    def chat_interface(user_input, history_messages=None):
        """
        Gradio界面的处理函数，使用openai-style messages 格式
        """
        if history_messages is None:
            history_messages = []

        if not user_input.strip():
            return "", history_messages, history_messages
        
        # 将用户输入添加到历史 (使用 messages 格式)
        history_messages.append({"role": "user", "content": user_input})
        
        # 获取AI回答 (确保 get_answer 支持 messages 格式的历史)
        ai_response = get_answer(user_input, history_messages)
        
        # 将AI回答添加到历史
        history_messages.append({"role": "assistant", "content": ai_response})
        
        # 返回值：清空输入框，更新聊天历史，更新状态
        return "", history_messages, history_messages

    # 创建Gradio界面
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🐧 Linux命令小助手")
        gr.Markdown("输入关于Linux命令的问题，我会根据官方文档为你解答。")
        
        # 使用Chatbot组件显示历史对话
        # 注意：Chatbot期望的格式是 [[msg1, msg2], ...]，其中msg1是用户，msg2是AI
        chatbot = gr.Chatbot(label="对话历史", type="messages", height=650)
        
        with gr.Row():
            with gr.Column(scale=4):
                user_input = gr.Textbox(placeholder="例如：如何查找一个文件？", label="你的问题", scale=4)
            with gr.Column(scale=1):
                submit_btn = gr.Button("发送", variant="primary", scale=1)
        
        # State组件现在存储元组格式的历史
        history_state = gr.State([]) 
        
        # 设置按钮点击事件，outputs顺序必须是 [user_input, chatbot, history_state]
        submit_btn.click(
            fn=chat_interface, 
            inputs=[user_input, history_state], 
            outputs=[user_input, chatbot, history_state] # 这里定义了输出顺序
        )
        # 也支持按回车键提交
        user_input.submit(
            fn=chat_interface, 
            inputs=[user_input, history_state], 
            outputs=[user_input, chatbot, history_state]
        )

    # 启动应用（仅当直接运行此脚本时）
    if __name__ == "__main__":
        print("\n🚀 启动Web界面...")
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

except ImportError:
    print("\n Gradio未安装。如需Web界面，请运行: pip install gradio")