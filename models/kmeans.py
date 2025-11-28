import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from models.utils import *
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)


class KmeansPetiClassifier:
    def __init__(self, anotated_dataset_file_path):
        logger.info("Iniciando objeto KmeansPetiClassifier")
        dataset = pd.read_csv(anotated_dataset_file_path)
        self._ai_utils = AIUtils()

        with open("./assets/stop_words.txt", "r", encoding="utf-8") as f:
            stop_words = f.read().splitlines()

        self._vectorizer = TfidfVectorizer(stop_words=stop_words)

        summarizes = dataset.summarize
        peti_types = dataset.peti_type.unique()

        random_state = 42  # TODO: load from dotenv

        X = self._vectorizer.fit_transform(summarizes.values)
        self._kmeans = KMeans(n_clusters=len(peti_types), random_state=random_state)
        self._kmeans.fit(X)
        self._cluster_labels = find_cluster_labels(
            dataset.peti_type.values, self._kmeans
        )

    def _classify_text(self, text_to_classify):
        logger.info(f"Classificando texto com K-Means")
        X_new = self._vectorizer.transform([text_to_classify])
        predict = self._kmeans.predict(X_new)[0]
        return self._cluster_labels[predict]

    def classify_peti(self, pdf_file_path):
        text = pdf_to_text(pdf_file_path)
        text_to_classify = self._ai_utils.summarize_text(text)
        pred = self._classify_text(text_to_classify)

        logger.info(f"Resultado K-Means: {pred}")

        return pred
