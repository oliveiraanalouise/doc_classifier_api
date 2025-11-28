import json
from models.utils import *
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EmbeddingPetiClassifier:
    def __init__(self, basedata_filename):
        logger.info("Iniciando objeto EmbeddingPetiClassifier")

        self._ai_utils = AIUtils()

        with open(basedata_filename, "r", encoding="utf-8") as f:
            database_peti_types = json.load(f)

        self._peti_types = list(database_peti_types.keys())

        # Embeddings da base de descrições para comparação
        self._embed_base = self._ai_utils.generate_embeddings_from_array(
            list(database_peti_types.values())
        )

    # Função para normalizar vetor
    def _normalize(self, v):
        vec = np.array(v)
        return vec / np.linalg.norm(vec)

    def _similarity_l2(self, target, base):
        target_vec = self._normalize(target).reshape(1, -1)
        base_matrix = np.array([self._normalize(b) for b in base])
        dists = np.linalg.norm(base_matrix - target_vec, axis=1)
        scores = -dists

        return scores

    def _similarity_cossine(self, target, base):
        target_vec = self._normalize(target).reshape(1, -1)
        base_matrix = np.array([self._normalize(b) for b in base])

        return cosine_similarity(target_vec, base_matrix)[0]

    def classify_peti_l2(self, pdf_file_path):
        text = pdf_to_text(pdf_file_path)
        sum_text = self._ai_utils.summarize_text(text)

        embed_file = self._ai_utils.generate_embedding_from_text(sum_text)

        sim_result = self._similarity_l2(
            embed_file.data[0].embedding, [e.embedding for e in self._embed_base.data]
        )
        result = self._peti_types[np.argmax(sim_result)]

        logger.info(f"Resultado embedding L2: {result}")

        return result

    def classify_peti_cossine(self, pdf_file_path):
        text = pdf_to_text(pdf_file_path)
        sum_text = self._ai_utils.summarize_text(text)

        embed_file = self._ai_utils.generate_embedding_from_text(sum_text)

        sim_result = self._similarity_cossine(
            embed_file.data[0].embedding, [e.embedding for e in self._embed_base.data]
        )
        result = self._peti_types[np.argmax(sim_result)]
        logger.info(f"Resultado embedding cossine: {result}")
        return result
