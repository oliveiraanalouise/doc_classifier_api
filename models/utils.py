from collections import Counter
import numpy as np
from openai import OpenAI
import ast
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import logging
from dotenv import load_dotenv
import os

load_dotenv(override=True)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AIUtils:
    def __init__(self):
        embed_api_key = os.getenv("EMBED_API_KEY")
        llm_api_key = os.getenv("LLM_API_KEY")

        # Cliente para acesso a API de embedding
        self._embed_client = OpenAI(
            api_key=embed_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )

        # Cliente para acesso a API de LLM
        self.llm_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=llm_api_key,
        )

    # Utiliza a LLM para fazer um resumo de um texto
    def summarize_text(self, text):
        logger.info(f"Resumindo texto através da LLM")

        # Configuração do prompt
        palavras = 200
        prompt = " Descreva, em aproximadamente {} palavras, em texto corrido, esse documento em português".format(
            palavras
        )

        # prompt LLM
        llm_response = self.llm_client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": text + prompt,
                }
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=False,
        )

        return llm_response.choices[0].message.content

    # Gera embeddings para um texto
    def generate_embedding_from_text(self, text):
        logger.info(f"Gerando embeddings a partir do texto")
        return self._embed_client.embeddings.create(
            input=[text],
            model="nvidia/nv-embedcode-7b-v1",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )

    def generate_embeddings_from_array(self, array_list):
        logger.info(f"Gerando embeddings a partir da lista")
        return self._embed_client.embeddings.create(
            input=array_list,
            model="nvidia/nv-embedcode-7b-v1",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )


def pdf_to_text(caminho_pdf: str) -> str:
    logger.info(f"Extraindo texto do pdf: {caminho_pdf}")
    reader = PdfReader(caminho_pdf)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() + "\n"
    return texto.strip()


def find_cluster_labels(y_labels, kmeans):
    cluster_labels = {}
    for c in range(kmeans.n_clusters):
        idx = np.where(kmeans.labels_ == c)[0]

        if len(idx) > 0:
            # rótulos manuais desses pontos
            labels_c = y_labels[idx]
            # rótulo predominante
            cluster_labels[c] = Counter(labels_c).most_common(1)[0][0]
        else:
            cluster_labels[c] = None

    return cluster_labels
