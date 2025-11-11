# build_vector_db.py (V2 - 使用阿里云 text-embedding-v3)
import os
import chromadb
from unstructured.partition.auto import partition
from dashscope import TextEmbedding
from dotenv import load_dotenv

# ========== 加载环境变量 ==========
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ DASHSCOPE_API_KEY 未设置！")

# ========== 配置项 ==========
KNOWLEDGE_BASE_DIR = "knowledge_base" # 知识库路径
CHROMA_DB_DIR = "db/chroma_db" # 向量数据库路径
COLLECTION_NAME = "linux_commands" # 数据库中的集合名称
# ====================================================

def load_and_chunk_documents():
    """
    从指定目录加载所有文档，并将其分割成小块。
    按每个命令条目进行分割，确保每个块包含完整的命令信息。
    """
    chunks = []
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if not os.path.isfile(file_path):
            continue
            
        try:
            elements = partition(filename=file_path)
            text = "\n".join([str(el) for el in elements])
            
            # 按命令条目分割，每个"-----------------------------------------------------------------"分隔符表示一个新命令
            command_sections = text.split('-----------------------------------------------------------------')
            
            # 处理每个命令条目
            for section in command_sections:
                # 清理空白字符
                section = section.strip()
                if not section:
                    continue
                    
                # 如果单个命令条目太长，进一步分割（限制在7500字符以内以确保安全余量）
                if len(section) > 7500:
                    # 按段落分割
                    paragraphs = section.split('\n\n')
                    current_chunk = ""
                    
                    for paragraph in paragraphs:
                        if len(current_chunk) + len(paragraph) < 7500:
                            current_chunk += paragraph + "\n\n"
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            # 如果单个段落就很长，进行句子级分割
                            if len(paragraph) > 7500:
                                # 按句子分割（以句号、感叹号、问号为分隔符）
                                sentences = []
                                current_sentence = ""
                                for char in paragraph:
                                    current_sentence += char
                                    if char in '.!?。！？' and len(current_sentence) > 100:
                                        sentences.append(current_sentence.strip())
                                        current_sentence = ""
                                if current_sentence:
                                    sentences.append(current_sentence.strip())
                                
                                # 组合句子到块中
                                current_sentence_chunk = ""
                                for sentence in sentences:
                                    if len(current_sentence_chunk) + len(sentence) < 7500:
                                        current_sentence_chunk += sentence + " "
                                    else:
                                        if current_sentence_chunk:
                                            chunks.append(current_sentence_chunk.strip())
                                        current_sentence_chunk = sentence + " "
                                
                                if current_sentence_chunk:
                                    chunks.append(current_sentence_chunk.strip())
                                current_chunk = ""
                            else:
                                current_chunk = paragraph + "\n\n"
                    
                    # 添加最后一个块
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                else:
                    # 长度适中的命令条目直接作为一个块
                    chunks.append(section)
                    
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue
            
    print(f"成功加载并分块了 {len(chunks)} 个文本片段。")
    return chunks

def get_embeddings(texts):
    """
    使用阿里云 text-embedding-v3 模型批量获取文本嵌入向量。
    """
    if not isinstance(texts, list):
        texts = [texts]
    
    # 检查并处理过长的文本，确保不超过8192字符限制
    processed_texts = []
    for text in texts:
        if len(text) > 8192:
            # 如果文本过长，截取前8192个字符并添加截断提示
            truncated_text = text[:8192] + " [内容已截断]"
            processed_texts.append(truncated_text)
            print(f"警告: 文本长度超过限制，已截断至8192字符")
        else:
            processed_texts.append(text)
        
    response = TextEmbedding.call(
        model='text-embedding-v3',
        input=processed_texts,
        api_key=DASHSCOPE_API_KEY
    )
    
    if response.status_code == 200:
        embeddings = [item['embedding'] for item in response.output['embeddings']]
        return embeddings
    else:
        raise Exception(f"嵌入API调用失败: {response.code}, {response.message}")

def setup_vector_database(chunks):
    """
    初始化ChromaDB，并使用阿里云嵌入模型将文本块存入。
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # 创建一个不带嵌入函数的集合，我们将手动提供嵌入
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 批量处理文本和嵌入，避免API调用次数过多
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        try:
            batch_embeddings = get_embeddings(batch_chunks)
            ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
            collection.add(
                ids=ids,
                embeddings=batch_embeddings, # 手动提供嵌入
                documents=batch_chunks
            )
            print(f"已处理 {min(i+batch_size, len(chunks))} / {len(chunks)} 个块")
        except Exception as e:
            print(f"处理批次 {i} 时出错: {e}")
            # 可以选择跳过或中断
            continue
    
    print(f"向量数据库已构建完成！")

if __name__ == "__main__":
    document_chunks = load_and_chunk_documents()
    if document_chunks:
        setup_vector_database(document_chunks)
        print("✅ 向量数据库构建成功！")
    else:
        print("⚠️ 没有找到任何文本块，请检查 knowledge_base 文件夹。")