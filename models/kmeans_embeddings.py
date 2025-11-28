import pandas as pd
from sklearn.cluster import KMeans
from models.utils import *
import logging
import ast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)


class KmeansEmbeddingPetiClassifier:
    def __init__(self, anotated_dataset_file_path):
        logger.info("Iniciando objeto KmeansEmbeddingPetiClassifier")
        dataset = pd.read_csv(anotated_dataset_file_path)
        self._ai_utils = AIUtils()
        dataset.embed_vec = dataset.embed_vec.apply(ast.literal_eval)
        embed_list_train = dataset.embed_vec.to_list()
        self._peti_types = dataset.peti_type.unique()

        random_state = 42  # TODO: load from dotenv

        self._kmeans = KMeans(
            n_clusters=len(self._peti_types), random_state=random_state
        )
        self._kmeans.fit(embed_list_train)

        self._cluster_labels = find_cluster_labels(
            dataset.peti_type.values, self._kmeans
        )

    def _classify_text(self, text_to_classify):
        sum_text = self._ai_utils.summarize_text(text_to_classify)
        embed_vec = (
            self._ai_utils.generate_embedding_from_text(sum_text).data[0].embedding
        )

        predict = self._kmeans.predict([embed_vec])[0]

        return self._cluster_labels[predict]

    def classify_peti(self, pdf_file_path):
        text = pdf_to_text(pdf_file_path)
        pred = self._classify_text(text)

        logger.info(f"Resultado K-Means & Embeddings: {pred}")

        return pred
