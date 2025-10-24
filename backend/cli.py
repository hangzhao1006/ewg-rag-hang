import os

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/apple/Downloads/ewg_data/secrets/llm-service-account.json"

import argparse
import pandas as pd
import json
import time
import glob
import hashlib
import chromadb
import numpy as np
import tempfile
from urllib.parse import urlparse
from sklearn.neighbors import NearestNeighbors


# Simple JSONL helpers used by EWG commands
def load_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, items):
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

# Vertex AI
from google import genai
from google.genai import types
from google.genai.types import Content, Part, GenerationConfig, ToolConfig
from google.genai import errors

# OpenAI fallback
import openai
try:
    # openai v1 exposes OpenAI class
    from openai import OpenAI as OpenAIClient
    OPENAI_HAS_NEW_API = True
except Exception:
    OpenAIClient = None
    OPENAI_HAS_NEW_API = False

# Langchain (support environments where langchain.text_splitter may not be present)
try:
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Some installations provide text splitters as a separate package
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from semantic_splitter import SemanticChunker
import agent_tools

# Setup
# GCP_PROJECT may be missing when the user prefers to use OpenAI APIs locally.
# Make it optional and only initialize the Vertex client when needed.
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = int(os.environ.get('EMBEDDING_DIMENSION', 256))
GENERATIVE_MODEL = "gemini-2.0-flash-001"
OPENAI_EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
OPENAI_CHAT_MODEL = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo')
INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"
# Allow overriding chroma host/port from environment when running on host vs inside docker
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "llm-rag-chromadb")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", 8000))

# If GCP service account JSON is stored in env var GCP_SA_KEY, write it to a temp file
def ensure_gcp_credentials_from_env() -> str | None:
        """Ensure ADC is available.

        Behavior:
        - If environment variable GCP_SA_KEY contains the JSON key (string), write it to
            a temporary file and set GOOGLE_APPLICATION_CREDENTIALS to that path.
        - Otherwise return whatever GOOGLE_APPLICATION_CREDENTIALS is set to (or None).
        """
        key = os.environ.get('GCP_SA_KEY')
        if key and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                tf = tempfile.NamedTemporaryFile('w', delete=False, suffix='.json')
                tf.write(key)
                tf.flush()
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = tf.name
                print('[INFO] Wrote GCP_SA_KEY to', tf.name)
                return tf.name
        return os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

#############################################################################
#                       Initialize the LLM Client                           #
llm_client = genai.Client(
    vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
# llm_client = None
# # Initialize Vertex AI client only when OpenAI key is not provided.
if not os.environ.get('OPENAI_API_KEY'):
    if not GCP_PROJECT:
        raise RuntimeError("GCP_PROJECT is not set and OPENAI_API_KEY is not provided. Set one of them to run the CLI.")
    llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
你是专注于 EWG（Environmental Working Group）护肤产品与成分信息的助手。仅使用提供的文本片段和元数据（title, brand, category, ingredients, warnings, ingredient concerns）来回答问题。不要使用外部知识或做超出片段信息的假设。

回答规则：
1. 只使用片段中的信息；若片段没有答案，明确说明“信息未提供”。
2. 汇总成分关注点时保留报告的级别（HIGH/MODERATE/LOW）并尽可能引用原文。
3. 若被问到产品“用途/功能”，优先使用记录的 `category` 字段。
4. 回答保持简洁、事实性，不进行臆造。
若多个片段信息冲突，请说明并展示冲突的片段。
"""

book_mappings = {
    
}


def generate_query_embedding(query):
    # If OPENAI_API_KEY is present prefer OpenAI embeddings (useful to match existing 1536-d collections)
    if os.environ.get('OPENAI_API_KEY'):
        key = os.environ.get('OPENAI_API_KEY')
        model = os.environ.get('OPENAI_EMBEDDING_MODEL', OPENAI_EMBEDDING_MODEL)
        if OPENAI_HAS_NEW_API and OpenAIClient is not None:
            client = OpenAIClient(api_key=key)
            resp = client.embeddings.create(model=model, input=query)
            return resp.data[0].embedding
        else:
            openai.api_key = key
            resp = openai.Embedding.create(model=model, input=query)
            return resp['data'][0]['embedding']

    # Default: use Vertex embeddings
    kwargs = {"output_dimensionality": EMBEDDING_DIMENSION}
    response = llm_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(**kwargs)
    )
    return response.embeddings[0].values


def find_buy_links_for_candidates(docs: list, metadatas: list | None, full_jsonl: str = "ewg_face_full.jsonl") -> dict:
    """给定检索到的文档文本和可选的 metadata，尝试在完整的 jsonl 数据集中查找对应的产品记录并返回购买链接字典。

    返回结构：{idx: [url1, url2, ...]}，idx 是 docs/metadatas 列表中的位置索引。
    匹配策略（逐步）：
      1. metadata 中的 source_url 或 url 精确匹配 full jsonl 的 url 字段；
      2. metadata 中的 title 与 full jsonl 的 title 相等或相互包含；
      3. 在文档文本中查找以 https:// 或 http:// 开头的 ewg 链接并匹配 full jsonl 的 url。
    这个函数是轻量的线性扫描（在文件较大时可能较慢），但避免修改现有索引流程，作为快速修复首选。
    """
    results = {i: [] for i in range(len(docs))}
    # 快速收集候选匹配值
    candidates = []
    for i, doc in enumerate(docs):
        cand = {
            'idx': i,
            'meta_url': None,
            'meta_title': None,
            'found_urls_in_doc': []
        }
        if metadatas and i < len(metadatas) and metadatas[i]:
            md = metadatas[i]
            cand['meta_url'] = md.get('source_url') or md.get('url') or md.get('source') or md.get('sourceUrl')
            cand['meta_title'] = md.get('title') or md.get('book')
        # find any http(s) urls in doc
        try:
            import re as _re
            urls = _re.findall(r"https?://[^\s'\)\"]+", doc)
            cand['found_urls_in_doc'] = urls
        except Exception:
            cand['found_urls_in_doc'] = []
        candidates.append(cand)

    # 如果 full jsonl 不存在，直接返回空
    if not os.path.exists(full_jsonl):
        return results

    def _is_purchase_link(u: str) -> bool:
        try:
            p = urlparse(u)
            domain = (p.netloc or '').lower()
            path = (p.path or '').lower()
            # 排除 site-internal links (ewg) 和 ingredient pages
            if 'ewg.org' in domain:
                # 如果是 ewg 的 product 页面，通常不是购买链接，排除之
                if '/skindeep/products/' in path:
                    return False
                # 排除 ingredient 页面
                if '/skindeep/ingredients/' in path:
                    return False
                # 排除常见站点导航/帮助页
                if any(x in path for x in ['/about', '/privacy', '/news', '/apps', '/skindeep/learn']):
                    return False
                return False
            # 排除社交媒体 / app store / tracking 等非购买目的域名
            black_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'youtube.com', 'play.google.com', 'apps.apple.com', 'linkedin.com']
            for bd in black_domains:
                if bd in domain:
                    return False
            # 排除 mailto/tel/javascript already handled earlier
            # 最终：接受外部域名作为可能的购买链接
            return bool(domain)
        except Exception:
            return False

    # 扫描 full jsonl 并匹配
    for rec in load_jsonl(full_jsonl):
        rec_url = rec.get('url') or rec.get('source_url') or ''
        rec_title = (rec.get('title') or '')
        # 优先使用 buy_button_urls/where_to_buy_urls
        buy_urls = rec.get('buy_button_urls') or rec.get('where_to_buy_urls') or []
        for cand in candidates:
            i = cand['idx']
            matched = False
            # 1) 精确 url 匹配
            if cand['meta_url'] and rec_url and cand['meta_url'].strip() == rec_url.strip():
                matched = True
            # 2) doc 内找到的 url 匹配
            if not matched and cand['found_urls_in_doc']:
                for u in cand['found_urls_in_doc']:
                    if rec_url and u.strip().startswith(rec_url.strip()):
                        matched = True
                        break
            # 3) title 包含匹配（case-insensitive）
            if not matched and cand['meta_title'] and rec_title:
                try:
                    if cand['meta_title'].strip().lower() in rec_title.strip().lower() or rec_title.strip().lower() in cand['meta_title'].strip().lower():
                        matched = True
                except Exception:
                    pass

            if matched and buy_urls:
                # 过滤并合并到结果，去重，限制数量（最多 5 个）
                count = 0
                for u in buy_urls:
                    if not u:
                        continue
                    if not _is_purchase_link(u):
                        continue
                    if u not in results[i]:
                        results[i].append(u)
                        count += 1
                    if count >= 5:
                        break

    return results


def generate_text_embeddings(chunks, dimensionality: int = 256, batch_size=250, max_retries=5, retry_delay=5):
    # If OPENAI_API_KEY is set, use OpenAI embeddings (batching supported)
    if os.environ.get('OPENAI_API_KEY'):
        key = os.environ.get('OPENAI_API_KEY')
        model = os.environ.get('OPENAI_EMBEDDING_MODEL', OPENAI_EMBEDDING_MODEL)
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    if OPENAI_HAS_NEW_API and OpenAIClient is not None:
                        client = OpenAIClient(api_key=key)
                        resp = client.embeddings.create(model=model, input=batch)
                        batch_emb = [item.embedding for item in resp.data]
                    else:
                        openai.api_key = key
                        resp = openai.Embedding.create(model=model, input=batch)
                        batch_emb = [d['embedding'] for d in resp['data']]
                    all_embeddings.extend(batch_emb)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"Failed to generate OpenAI embeddings after {max_retries} attempts. Last error: {e}")
                        raise
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    print(f"OpenAI embedding error: {e}. Retrying in {wait_time}s (attempt {retry_count})")
                    time.sleep(wait_time)
        return all_embeddings

    # Default: use Vertex embeddings
    # Max batch size is 250 for Vertex AI
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Retry logic with exponential backoff
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = llm_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=dimensionality),
                )
                all_embeddings.extend(
                    [embedding.values for embedding in response.embeddings])
                break

            except errors.APIError as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(
                        f"Failed to generate embeddings after {max_retries} attempts. Last error: {str(e)}")
                    raise

                # Calculate delay with exponential backoff
                wait_time = retry_delay * (2 ** (retry_count - 1))
                print(
                    f"API error (code: {e.code}): {e.message}. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                time.sleep(wait_time)

    return all_embeddings


def load_text_embeddings(df, collection, batch_size=500):

    # Generate ids
    df["id"] = df.index.astype(str)
    hashed_books = df["book"].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    df["id"] = hashed_books + "-" + df["id"]

    metadata = {
        "book": df["book"].tolist()[0]
    }
    if metadata["book"] in book_mappings:
        book_mapping = book_mappings[metadata["book"]]
        metadata["author"] = book_mapping["author"]
        metadata["year"] = book_mapping["year"]

    # Process data in batches
    total_inserted = 0
    for i in range(0, df.shape[0], batch_size):
        # Create a copy of the batch and reset the index
        batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

        ids = batch["id"].tolist()
        documents = batch["chunk"].tolist()
        metadatas = [metadata for item in batch["book"].tolist()]
        embeddings = batch["embedding"].tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} items...")

    print(
        f"Finished inserting {total_inserted} items into collection '{collection.name}'")


def chunk(method="char-split"):
    print("chunk()")

    # Make dataset folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get the list of text file
    text_files = glob.glob(os.path.join(INPUT_FOLDER, "books", "*.txt"))
    print("Number of files to process:", len(text_files))

    # Process
    for text_file in text_files:
        print("Processing file:", text_file)
        filename = os.path.basename(text_file)
        book_name = filename.split(".")[0]

        with open(text_file) as f:
            input_text = f.read()

        text_chunks = None
        if method == "char-split":
            chunk_size = 350
            chunk_overlap = 20
            # Init the splitter
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator='', strip_whitespace=False)

            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "recursive-split":
            chunk_size = 350
            # Init the splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size)

            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "semantic-split":
            # Init the splitter
            text_splitter = SemanticChunker(
                embedding_function=generate_text_embeddings)
            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])

            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        if text_chunks is not None:
            # Save the chunks
            data_df = pd.DataFrame(text_chunks, columns=["chunk"])
            data_df["book"] = book_name
            print("Shape:", data_df.shape)
            print(data_df.head())

            jsonl_filename = os.path.join(
                OUTPUT_FOLDER, f"chunks-{method}-{book_name}.jsonl")
            with open(jsonl_filename, "w") as json_file:
                json_file.write(data_df.to_json(orient='records', lines=True))


def embed(method="char-split"):
    print("embed()")

    # Get the list of chunk files
    jsonl_files = glob.glob(os.path.join(
        OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    # Process
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        chunks = data_df["chunk"].values
        chunks = chunks.tolist()
        if method == "semantic-split":
            embeddings = generate_text_embeddings(
                chunks, EMBEDDING_DIMENSION, batch_size=15)
        else:
            embeddings = generate_text_embeddings(
                chunks, EMBEDDING_DIMENSION, batch_size=100)
        data_df["embedding"] = embeddings

        time.sleep(5)

        # Save
        print("Shape:", data_df.shape)
        print(data_df.head())

        jsonl_filename = jsonl_file.replace("chunks-", "embeddings-")
        with open(jsonl_filename, "w") as json_file:
            json_file.write(data_df.to_json(orient='records', lines=True))


def load(method="char-split"):
    print("load()")

    # Clear Cache
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"
    print("Creating collection:", collection_name)

    try:
        # Clear out any existing items in the collection
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        print(f"Collection '{collection_name}' did not exist. Creating new.")

    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})
    print(f"Created new empty collection '{collection_name}'")
    print("Collection:", collection)

    # Get the list of embedding files
    jsonl_files = glob.glob(os.path.join(
        OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    # Process
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        # Load data
        load_text_embeddings(data_df, collection)


def query(method="char-split", user_query: str | None = None):
    print("load()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"

    query = user_query if user_query else "有没有保湿产品推荐？"
    query_embedding = generate_query_embedding(query)
    print("Embedding values:", query_embedding)

    # Get the collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Collection '{collection_name}' not found or Chroma error: {e}")
        print("Make sure you've run --load to create and populate the collection first.")
        return

    # # 1: Query based on embedding value
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # # 2: Query based on embedding value + metadata filter
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10,
    # 	where={"book":"The Complete Book of Cheese"}
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # # 3: Query based on embedding value + lexical search filter
    # search_string = "Italian"
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10,
    # 	where_document={"$contains": search_string}
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # Perform a simple embedding-based query and print results
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=10)
        print("Query:", query)
        # Print raw results dict for debugging
        # print("\n\nResults:", results)

        # Pretty-print top documents with metadata and (if available) distances
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else None
        distances = results.get("distances", [[]])[0] if results.get("distances") else None

        print("\n--- Top results ---")
        if not docs:
            print("No documents returned for this query.")
        for i, doc in enumerate(docs):
            md = metadatas[i] if metadatas else {}
            dist = distances[i] if distances else None
            print(f"[{i}] distance={dist} metadata={md}")
            print(doc[:500].replace('\n', ' '), "...\n")

        # 查找并打印每个检索到的文档对应的购买链接（若存在）
        try:
            buy_map = find_buy_links_for_candidates(docs, metadatas, full_jsonl="ewg_face_full.jsonl")
            for idx, links in buy_map.items():
                if links:
                    print(f"[BUY LINKS] doc[{idx}] -> {links}")
        except Exception as e:
            print('[BUY LOOKUP] error:', repr(e))

    except Exception as e:
        print('Query failed:', repr(e))


def chat(method="char-split", user_query: str | None = None):
    print("chat()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"

    query = user_query if user_query else "How is cheese made?"
    query_embedding = generate_query_embedding(query)
    print("Query:", query)
    print("Embedding values:", query_embedding)
    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Query based on embedding value
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    print("\n\nResults:", results)

    print(len(results["documents"][0]))

    INPUT_PROMPT = f"""
    {query}
    {"\n".join(results["documents"][0])}
    """

    print("INPUT_PROMPT: ", INPUT_PROMPT)

    # 查找并附加购买链接到 prompt（使 LLM 能直接看到链接）
    try:
        docs = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else None
        buy_map = find_buy_links_for_candidates(docs, metadatas, full_jsonl="ewg_face_full.jsonl")
        extra_lines = []
        for i, links in buy_map.items():
            if links:
                extra_lines.append(f"[BUY_LINKS_{i}]: " + ", ".join(links))
        if extra_lines:
            INPUT_PROMPT = INPUT_PROMPT + "\n\nPurchase links found:\n" + "\n".join(extra_lines)
            print("APPENDED BUY LINKS TO PROMPT:", extra_lines)
    except Exception as e:
        print('[BUY LOOKUP] chat append error:', repr(e))

    # If OPENAI_API_KEY is present, prefer OpenAI Chat for more human-like replies
    if os.environ.get('OPENAI_API_KEY'):
        key = os.environ.get('OPENAI_API_KEY')
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION.strip()},
            {"role": "user", "content": INPUT_PROMPT}
        ]
        try:
            if OPENAI_HAS_NEW_API and OpenAIClient is not None:
                client = OpenAIClient(api_key=key)
                resp = client.chat.completions.create(model=OPENAI_CHAT_MODEL, messages=messages, temperature=0.0)
                # new client returns choices with messages/text
                # try to extract text
                generated_text = ''
                for choice in resp.choices:
                    if getattr(choice, 'message', None) and getattr(choice.message, 'content', None):
                        generated_text += choice.message.content
                    elif getattr(choice, 'delta', None):
                        generated_text += getattr(choice.delta, 'content', '')
                print('LLM Response (OpenAI):', generated_text)
            else:
                openai.api_key = key
                resp = openai.ChatCompletion.create(model=OPENAI_CHAT_MODEL, messages=messages, temperature=0.0)
                generated_text = resp['choices'][0]['message']['content'].strip()
                print('LLM Response (OpenAI):', generated_text)
        except Exception as e:
            print('[CHAT] OpenAI generation error:', repr(e))
        return

    # Fallback to Vertex AI generation
    try:
        response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL, contents=INPUT_PROMPT
        )
        generated_text = response.text
        print('LLM Response (Vertex):', generated_text)
    except Exception as e:
        print('[CHAT] Vertex generation error:', repr(e))


###############################
# EWG-specific simplified CLI
###############################

def ewg_chunk(in_jsonl: str, out_chunks: str, chunk_size: int = 800, chunk_stride: int = 200):
    print('[EWG] chunking', in_jsonl, '->', out_chunks)
    items = []
    for rec in load_jsonl(in_jsonl):
        # build canonical text similar to parse_label_info.make_chunks_from_record
        parts = []
        if rec.get('title'):
            parts.append(f"Title: {rec.get('title')}")
        if rec.get('brand'):
            parts.append(f"Brand: {rec.get('brand')}")
        if rec.get('category'):
            parts.append(f"Category: {rec.get('category')}")
        ls = rec.get('label_sections', {})
        for k in ['ingredients','directions','warnings']:
            v = ls.get(k, {}).get('text') if ls.get(k) else None
            if v:
                parts.append(f"{k.capitalize()}: {v}")
        if rec.get('ingredient_concern'):
            parts.append(f"Ingredient concerns: {rec.get('ingredient_concern')}")
        text = '\n'.join(parts).strip()
        if not text:
            continue
        L = len(text)
        start = 0
        while start < L:
            end = min(start + chunk_size, L)
            chunk_text = text[start:end].strip()
            if chunk_text:
                items.append({'text': chunk_text, 'source_url': rec.get('url'), 'title': rec.get('title')})
            if end == L:
                break
            start += chunk_size - chunk_stride
    write_jsonl(out_chunks, items)
    print('[EWG] wrote', len(items), 'chunks')


def ewg_embed(chunks_jsonl: str, out_index: str, out_meta: str):
    # Ensure gcp creds if provided as secret
    ensure_gcp_credentials_from_env()
    chunks = list(load_jsonl(chunks_jsonl))
    texts = [c['text'] for c in chunks]
    print('[EWG] embedding', len(texts), 'chunks')
    # use existing generate_text_embeddings (Vertex or OpenAI based on env)
    embeddings = generate_text_embeddings(texts, EMBEDDING_DIMENSION)
    arr = np.array(embeddings, dtype=np.float32)

    # If using OpenAI, avoid overwriting Vertex file by default
    if os.environ.get('OPENAI_API_KEY') and out_index.endswith('.npz'):
        out_index_openai = out_index.replace('.npz', '_openai.npz')
    else:
        out_index_openai = out_index

    np.savez(out_index_openai, embeddings=arr)
    write_jsonl(out_meta, chunks)
    print('[EWG] saved index', out_index_openai, 'and meta', out_meta)


def ewg_query(index_npz: str, metadata_jsonl: str, question: str, top_k: int = 5):
    data = np.load(index_npz, allow_pickle=True)
    embeddings = data['embeddings']
    meta = list(load_jsonl(metadata_jsonl))
    nn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
    nn.fit(embeddings)
    # generate query embedding
    q_emb = generate_text_embeddings([question], EMBEDDING_DIMENSION)[0]
    ids = nn.kneighbors([q_emb], n_neighbors=top_k, return_distance=False)[0]
    contexts = [meta[i]['text'] for i in ids]
    print('\n--- Top contexts ---')
    for i,c in enumerate(contexts):
        print(f'[{i}]', c[:400].replace('\n',' '), '...')
    # attempt answer generation via Vertex
    # ensure_gcp_credentials_from_env()
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    # If OPENAI_API_KEY is provided, use OpenAI Chat for generation
    if os.environ.get('OPENAI_API_KEY'):
        print('[EWG] OPENAI_API_KEY present, attempting to generate via OpenAI Chat')
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        # Build prompt using system instruction + question + contexts
        prompt = SYSTEM_INSTRUCTION.strip() + "\n\nQuestion: " + question + "\n\nRetrieved contexts:\n" + "\n\n".join([f"[{i}] {c}" for i, c in enumerate(contexts)])
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION.strip()},
                    {"role": "user", "content": "Question: " + question + "\n\nRetrieved contexts:\n" + "\n\n".join([f"[{i}] {c}" for i, c in enumerate(contexts)])}
                ],
                temperature=0.0,
                max_tokens=512
            )
            answer = resp['choices'][0]['message']['content'].strip()
            print('\n--- Generated answer (OpenAI) ---\n')
            print(answer)
        except Exception as e:
            print('[EWG] OpenAI generation error:', repr(e))
        return
    if creds_path:
        print('[EWG] GCP creds present, attempting to generate via Vertex (if configured)')
        # Prepare prompt for generation: include system instruction, user question, and retrieved contexts
        try:
            # Combine system instruction, user question and retrieved contexts into a single user content
            contexts_text = "\n\n".join([f"[{i}] {c}" for i, c in enumerate(contexts)])
            user_text = SYSTEM_INSTRUCTION.strip() + "\n\nQuestion: " + question + "\n\nRetrieved contexts:\n" + contexts_text
            user_content = Content(role="user", parts=[Part(text=user_text)])

            response = llm_client.models.generate_content(
                model=GENERATIVE_MODEL,
                contents=[user_content],
                config=types.GenerateContentConfig(temperature=0.0)
            )

            # Collect text from response candidates
            generated = []
            for cand in response.candidates:
                for part in cand.content.parts:
                    if getattr(part, 'text', None):
                        generated.append(part.text)

            answer = "\n".join(generated).strip()
            print('\n--- Generated answer ---\n')
            if answer:
                print(answer)
            else:
                print('[EWG] Vertex returned no text in candidates; full response:')
                print(response)

        except errors.APIError as e:
            print('[EWG] Vertex generation APIError:', e)
            # print details if available
            try:
                print('Code:', e.code)
                print('Message:', e.message)
            except Exception:
                pass
        except Exception as e:
            print('[EWG] Vertex generation unexpected error:', repr(e))
    else:
        print('[EWG] No GCP creds; please set OPENAI_API_KEY to generate an answer via OpenAI')


def install_gcp_key(src_path: str):
    """Copy a local GCP service account JSON into .secrets/gcp_sa.json and set env var."""
    os.makedirs('.secrets', exist_ok=True)
    dst = os.path.join('.secrets', 'gcp_sa.json')
    with open(src_path, 'r', encoding='utf-8') as sf, open(dst, 'w', encoding='utf-8') as df:
        df.write(sf.read())
    print('[INFO] Copied key to', dst)
    print('Add the following to your environment to use it:')
    print(f"export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/{dst}")
    return dst



def get(method="char-split"):
    print("get()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Get documents with filters
    results = collection.get(
        where={"book": "The Complete Book of Cheese"},
        limit=10
    )
    print("\n\nResults:", results)


def agent(method="char-split"):
    print("agent()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"
    # Get the collection
    collection = client.get_collection(name=collection_name)

    # User prompt
    user_prompt_content = Content(
        role="user",
        parts=[
            Part(text="Describe where cheese making is important in Pavlos's book?"),
        ],
    )

    # Step 1: Prompt LLM to find the tool(s) to execute to find the relevant chunks in vector db
    print("user_prompt_content: ", user_prompt_content)
    response = llm_client.models.generate_content(
        model=GENERATIVE_MODEL,
        contents=user_prompt_content,
        config=types.GenerateContentConfig(
            temperature=0,
            tools=[agent_tools.cheese_expert_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="any"
                )
            )
        )
    )
    print("LLM Response:", response)

    # Step 2: Execute the function and send chunks back to LLM to answer get the final response
    function_calls = [part.function_call for part in response.candidates[0].content.parts if part.function_call]
    print("Function calls:", function_calls)
    function_responses = agent_tools.execute_function_calls(
        function_calls, collection, embed_func=generate_query_embedding)
    if len(function_responses) == 0:
        print("Function calls did not result in any responses...")
    else:
        # Call LLM with retrieved responses
        response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=[
                user_prompt_content,  # User prompt
                response.candidates[0].content,  # Function call response
                Content(
                    parts=function_responses
                ),
            ],
            config=types.GenerateContentConfig(
                tools=[agent_tools.cheese_expert_tool]
            )
        )
        print("LLM Response:", response)


def main(args=None):
    print("CLI Arguments:", args)

    if args.chunk:
        chunk(method=args.chunk_type)

    if args.embed:
        embed(method=args.chunk_type)

    if args.load:
        load(method=args.chunk_type)

    if args.query:
        query(method=args.chunk_type, user_query=args.query_text)

    if args.chat:
        chat(method=args.chunk_type, user_query=args.query_text)

    if args.get:
        get(method=args.chunk_type)

    if args.agent:
        agent(method=args.chunk_type)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Chunk text",
    )
    parser.add_argument(
        "--ewg-chunk",
        nargs=2,
        metavar=("IN","OUT"),
        help="Chunk an EWG JSONL: provide input jsonl and output chunks jsonl",
    )
    parser.add_argument(
        "--ewg-chunk-size",
        type=int,
        default=800,
        help="Chunk size for --ewg-chunk (default 800)",
    )
    parser.add_argument(
        "--ewg-chunk-stride",
        type=int,
        default=200,
        help="Chunk stride for --ewg-chunk (default 200)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate embeddings",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load embeddings to vector db",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Query vector db",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat with LLM",
    )
    parser.add_argument(
        "--get",
        action="store_true",
        help="Get documents from vector db",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Chat with LLM Agent",
    )
    parser.add_argument("--chunk_type", default="char-split",
                        help="char-split | recursive-split | semantic-split")
    parser.add_argument(
        "--query-text",
        dest="query_text",
        default=None,
        help="Provide a custom question text to use with --query or --chat",
    )
    parser.add_argument("--install-gcp-key", default=None,
                        help="Copy a local GCP service account JSON into workspace .secrets and print export command")

    args = parser.parse_args()

    # If requested, install the GCP key before running other actions
    if args.install_gcp_key:
        install_gcp_key(args.install_gcp_key)

    # Handle EWG-specific chunk before other actions (convenience wrapper)
    if args.ewg_chunk:
        in_path, out_path = args.ewg_chunk
        print('[CLI] Running ewg_chunk', in_path, '->', out_path)
        ewg_chunk(in_path, out_path, chunk_size=args.ewg_chunk_size, chunk_stride=args.ewg_chunk_stride)

    main(args)
