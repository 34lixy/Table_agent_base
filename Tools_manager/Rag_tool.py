import concurrent.futures
import os
import pickle
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from .file_process import process_path


class RAGService:
    def __init__(self,
                 model_path='D:/work/中电信AI/model/bge-small-zh-v1.5',
                 device='cpu',
                 chunk_size=1000, chunk_overlap=200,
                 embedding_cls=HuggingFaceBgeEmbeddings,
                 text_splitter_cls=RecursiveCharacterTextSplitter):

        model_config = {"device": device}
        embedding_config = {"normalize_embeddings": True}
        self.embedding_model = embedding_cls(
            model_name=model_path, model_kwargs=model_config, encode_kwargs=embedding_config
        )
        self.text_splitter = text_splitter_cls(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    def load_and_split_documents(self, input_paths, chunk_nums):
        documents = []

        for path in input_paths:
            try:
                if path.endswith(('.csv', '.xlsx', '.txt', '.json')):
                    doc = process_path(path, self.text_splitter, chunk_nums)
                    documents.extend(doc)
                else:
                    print(f"Unsupported path type for path: {path}")

            except Exception as e:
                print(f"Failed to process path {path}: {e}")

        return documents

    def initialize_document_vector(self, document):
        vector_store = FAISS.from_documents(document, embedding=self.embedding_model)
        return vector_store

    def initialize_vector_store(self, input_paths, chunk_nums=None):
        if isinstance(input_paths, str):
            input_paths = [input_paths]

        document = self.load_and_split_documents(input_paths, chunk_nums)

        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        doc_file_path = os.path.join(cache_dir, 'document.pkl')
        with open(doc_file_path, 'wb') as f:
            pickle.dump(document, f)
        try:
            if not chunk_nums:
                vector_store = self.initialize_document_vector(document)

            else:
                print("Initializing vector store with %s parts...",
                      chunk_nums if chunk_nums else "all documents as one part")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    vector_store = list(executor.map(self.initialize_document_vector, document))

            file_path = os.path.join(cache_dir, 'vectors_store.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(vector_store, f)
        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            raise


class SimilaritySearcher:
    def __init__(self,
                 document_path='cache/document.pkl',
                 vectors_single_path='cache/vectors_store.pkl'):

        with open(document_path, 'rb') as f:
            self.document = pickle.load(f)
        with open(vectors_single_path, 'rb') as f:
            self.vector_store = pickle.load(f)

        self.byte_store = InMemoryByteStore()
        self.document_key = "doc_id"

        self.multi_retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            byte_store=self.byte_store,
            id_key=self.document_key,
        )

    def retrieve_similar_documents(self, query, vector_store, document, chunk_nums=None):
        """检索与 query 相关的相似文档."""

        try:
            document_ids = [str(uuid.uuid4()) for _ in document]
            self.multi_retriever.docstore.mset(list(zip(document_ids, document)))

            if chunk_nums:
                similar_chunks = vector_store.similarity_search_with_relevance_scores(query, k=3)
                return [{"content": doc[0].page_content.replace('\n', ', '), "score": doc[1]} for doc in similar_chunks]
            return vector_store.similarity_search(query, k=3)
        except Exception as e:
            print(f"Error during similarity search for query '{query}': {e}")
            return []

    def process_single_query(self, query, chunk_nums):
        """处理单个 query 并返回相似文档."""

        # 从单个 vector_store 检索
        if not chunk_nums:
            context = self.retrieve_similar_documents(query, self.vector_store, self.document)
            return "\n------------\n".join(doc.page_content.replace('\n', ', ') for doc in context)

        # 从多个 vector_store 检索，按照相似度排序并返回结果
        context_list = []
        for vector, doc in zip(self.vector_store, self.document):
            context = self.retrieve_similar_documents(query, vector, doc, chunk_nums)
            context_list.extend(context)
        sorted_list = sorted(context_list, key=lambda x: x['score'], reverse=True)
        return "\n------------\n".join(doc['content'] for doc in sorted_list)

    def process_queries(self, queries, chunk_nums=None):
        """处理多个 query，复用初始化好的向量存储."""
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for query in queries:
            result = self.process_single_query(query, chunk_nums)
            # print(f"Performing similarity search result of the  {query} :\n {result}")
            results.append(result)
        return results


def retriever_tool(query: list):
    """
    '''query:问题列表
    若模型无法通过已有知识回答用户问题时,优先考虑采用此工具而不是搜索引擎,从本地知识库中回答用户问题,需要提供用户问题参数
    """
    searcher = SimilaritySearcher()
    res = searcher.process_queries(query)
    return res


if __name__ == '__main__':
    rag = RAGService()
    files_paths = ['../data/最近10年行政区数.csv']
    questions = ["请理解给出数据源中的指标名及对应的数字，输出2023年的县级市数", '2023年县级市数']
    rag.initialize_vector_store(files_paths, chunk_nums=3)


    searcher = SimilaritySearcher()
    res = searcher.process_queries(questions, chunk_nums=3)
    print(res)
