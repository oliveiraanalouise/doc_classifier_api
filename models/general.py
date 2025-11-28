from google import genai
from google.genai import types
import pathlib
import sys
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from models.utils import *
from dotenv import load_dotenv
import os

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Iniciando objeto GeneralClassifier")


class GeneralClassifier:
    def __init__(self, general_base_file_path):
        self._ai_utils = AIUtils()

        self._client = genai.Client(api_key=os.getenv("GENERAL_API_KEY"))

        with open(general_base_file_path, "r", encoding="utf-8") as f:
            self._dataset = json.load(f)

        self._keys = self._dataset.keys()

    def _normalize(self, v):
        vec = np.array(v.values)
        return vec / np.linalg.norm(vec)

    def _similarity_cossine(self, target, base):
        normalized = []
        for t in target:
            normalized.append(self._normalize(t))

        for b in base:
            normalized.append(self._normalize(b))

        return cosine_similarity(np.array(normalized))

    def _similarity_l2(self, target, base):
        normalized = []
        for t in target:
            normalized.append(self._normalize(t))

        for b in base:
            normalized.append(self._normalize(b))

        return euclidean_distances(np.array(normalized))

    def classify_doc_cossine(self, doc_file_path):
        doc_data = pathlib.Path(doc_file_path)

        tokens = 50
        prompt = (
            "Descreva, em no máximo {} palavras, esse documento em português".format(
                tokens
            )
        )
        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=doc_data.read_bytes(), mime_type="application/pdf"
                ),
                prompt,
            ],
        )

        config = types.EmbedContentConfig(
            output_dimensionality=768, task_type="SEMANTIC_SIMILARITY"
        )
        embbed_base = self._client.models.embed_content(
            model="gemini-embedding-001", contents=self._dataset.values(), config=config
        )

        embbed = self._client.models.embed_content(
            model="gemini-embedding-001", contents=response.text, config=config
        )

        sim_result = self._similarity_cossine(embbed.embeddings, embbed_base.embeddings)
        base_list = list(self._keys)
        result = base_list[np.argmax(sim_result[0][1:])]
        logger.info(f"Resultado general Cossine: {result}")

        return result

    def classify_doc_l2(self, doc_file_path):
        doc_data = pathlib.Path(doc_file_path)

        tokens = 50
        prompt = (
            "Descreva, em no máximo {} palavras, esse documento em português".format(
                tokens
            )
        )
        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=doc_data.read_bytes(), mime_type="application/pdf"
                ),
                prompt,
            ],
        )

        config = types.EmbedContentConfig(
            output_dimensionality=768, task_type="SEMANTIC_SIMILARITY"
        )
        embbed_base = self._client.models.embed_content(
            model="gemini-embedding-001", contents=self._dataset.values(), config=config
        )

        embbed = self._client.models.embed_content(
            model="gemini-embedding-001", contents=response.text, config=config
        )

        sim_result = self._similarity_l2(embbed.embeddings, embbed_base.embeddings)

        base_list = list(self._keys)
        result = base_list[np.argmax(sim_result[0][1:])]
        logger.info(f"Resultado general L2: {result}")

        return result
