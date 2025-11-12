# app.py
import os
import re
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
# 解析提问
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

# 检索数据库
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

# 获取模型回答
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

# 获取判断是否需要检索的提问词
def get_retrieve_prompt(user_question):
    return f"""你是一个严格的Linux命令知识库访问控制器。
请根据以下用户问题，严格判断是否需要从Linux命令知识库中检索信息。
只允许在以下情况回答"需要"：
1. 用户明确询问某个Linux命令的具体用法、参数、作用
2. 用户问题中包含"命令"、"指令"、"用法"、"语法"等关键词
3. 用户直接提及具体的Linux命令名称（如su, ls, grep等）

在以下情况必须回答"不需要"：
1. 用户在设置昵称、问候、告别等社交互动
2. 用户问题与Linux命令无关
3. 用户只是泛泛而谈"Linux"但未涉及具体命令
    
用户问题{user_question}
    
请严格按照要求回答，只能回答"需要"或"不需要"，不能添加其他内容。
"""

# 判断是否需要从数据库中检索信息
def is_need_retrieve(user_question):
    """
    通过多重判断来决定是否需要去检索数据库
    判断一：检查是否包含Linux命令关键词
    判断二：检查是否包含"命令"、"指令"、"用法"、"语法"、"参数"等关键词
    判断三：检查是否存在社交或设置请求
    """

    # 设置硬性规则过滤
    user_question = user_question.strip().lower()

    result = None # 默认为空

    # 判断一：检查是否包含Linux命令关键词
    linux_commands = [
    "man", "help", "info", "shutdown", "reboot", "halt", "poweroff", "pwd", 
    "cd", "tree", "mkdir", "touch", "ls", "cp", "mv", "rm", "rmdir", "ln", 
    "readlink", "find", "xargs", "rename", "basename", "dirname", "chattr",
    "lsattr", "file", "md5sum", "chown", "chmod", "chgrp", "umask", "cat",
    "tac", "more", "less", "head", "tail", "tailf", "cut", "split", "paste",
    "sort", "join", "uniq", "wc", "iconv", "dos2unix", "diff", "vimdiff", 
    "rev", "tr", "od", "tee", "vi", "vim", "grep", "sed", "awk", "uname", 
    "hostname", "demsg", "stat", "du", "date", "echo", "watch", "which", 
    "whereis", "locate", "updatedb", "tar", "gzip", "zip", "unzip", "scp", 
    "rsync", "useradd", "usermod", "userdel", "groupadd", "groupdel", "passwd", 
    "chage", "chpasswd", "su", "visudo", "sudo", "id", "w", "who", "users", 
    "whoami", "last", "lastb", "latslog", "fdisk", "partprobe", "tune2fs", 
    "parted", "mkfs", "dumpe2fs", "resize2fs", "fsck", "dd", "mount", 
    "umount", "df", "mkswap", "swapon", "swapoff", "sync", "ps", 
    "pstree", "pgrep", "kill", "killall", "pkill", "top", "nice", 
    "renice", "nohup", "strace", "ltrace", "runlevel", "init", "service",
    "ifconfig", "ifup", "ifdown", "route", "arp", "ip", "netstat", "ss", 
    "ping", "traceroute", "arping", "telnet", "curl", "nc", "ssh", "wget", 
    "mail", "mailq", "nslookup", "dig", "host", "nmap", "tcpdump", "lsof", 
    "uptime", "free", "iftop", "vmstat", "mpstat", "iostat", "iotop", "sar", 
    "chkconfig", "ntsysv", "setup", "ethtool", "mii-tool", "dmidecode", 
    "lspci", "ipcs", "ipcrm", "rpm", "yum", ":", "source", "test", "alias", 
    "unalias", "bg", "fg", "jobs", "break", "continue", "eval", "exit", 
    "logout", "export", "history", "read", "type", "ulimit", "unset"]

    words = re.findall(r'\b[a-zA-Z]+\b', user_question) # 用正则表达式提取所有单词
    if any(cmd in words for cmd in linux_commands):
        return true

    # 判断二：检查是否包含"命令"、"指令"、"用法"、"语法"等关键词
    keywords = ["命令", "指令", "用法", "语法", "参数"]
    if any(keyword in user_question for keyword in keywords):
        return True
    
    # 判断三：检查是否存在社交或设置请求
    social_keywords = ["你好", "称呼", "名字", "hi", "hello", "早上好", "晚上好", "再见", "bye"]
    if any(keyword in user_question for keyword in social_keywords):
        return False
    
    # 无法判断则返回None，交给大模型判断
    return None

# 获取答案
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

    # ============ 2. 判断是否需要检索数据库 ============

    # 调用大模型进行判断
    judge_response = is_need_retrieve(user_question)

    # 在需要时检索数据库
    if judge_response == True:
        contexts = retrieve_context(user_question, n_results=2)
    elif judge_response == False:
        contexts = []
    else:
        #获取提示词
        judge_prompt = get_retrieve_prompt(user_question)
        judge_response2 = call_qwen_api(judge_prompt)
        if "需要" in judge_response2:
            contexts = retrieve_context(user_question, n_results=2)
        else:
            contexts = []

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
7. 保持教育性和实用性，帮助用户真正理解命令的使用方法和注意事项
8. 遵守用户在历史对话中提出的要求
【当前问题】
{user_question}

现在请基于知识库内容给出专业解答："""
    else:
        if judge_response == None:
            retrieve_prompt = "结果检索，知识库中没有找到与用户问题相关的具体信息。"
        else:
            retrieve_prompt = "用户的问题似乎与Linux命令无关，将基于通用知识进行回答。"
        # 没有知识库信息：使用通用问答模式
        final_prompt = f"""你是一个专业的Linux命令学习助手，专门为用户学习Linux命令提供帮助。{retrieve_prompt}
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

现在请基于你的通用Linux知识给出专业解答，并在开头说明"{retrieve_prompt}以下基于通用Linux知识解答："："""

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
        
        # 将 history_messages 从字典格式转换为元组格式
        history_tuples = []
        for i in range(0, len(history_messages), 2):
            if i + 1 < len(history_messages):
                user_msg = history_messages[i]["content"]
                ai_msg = history_messages[i+1]["content"]
                history_tuples.append((user_msg, ai_msg))
        
        # 获取AI回答 (确保 get_answer 支持 messages 格式的历史)
        ai_response = get_answer(user_input, history_tuples)
        
        # 将用户输入和AI回答添加到历史
        history_messages.append({"role": "user", "content": user_input})
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
