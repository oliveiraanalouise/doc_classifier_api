from flask import Flask, request
from flask_restx import Resource, Api

from models.kmeans import KmeansPetiClassifier
from models.kmeans_embeddings import KmeansEmbeddingPetiClassifier
from models.embeddings import EmbeddingPetiClassifier
from models.general import GeneralClassifier

import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./uploads"


# TODO: load from dotenv
anotated_dataset_file_path = "./assets/df_manual_anot.csv"
base_data_types_file_path = "./assets/base_tipos_peticoes_v1_embeddings.json"
general_base_file_path = "./assets/base_dados_general.json"

general_classifier = GeneralClassifier(general_base_file_path)
kmeans_peti_classifier = KmeansPetiClassifier(anotated_dataset_file_path)
kmeans_embed_peti_classifier = KmeansEmbeddingPetiClassifier(anotated_dataset_file_path)
embedding_peti_classifier = EmbeddingPetiClassifier(base_data_types_file_path)


api = Api(
    app,
    version="0.1",
    title="AI Doc Classifier",
    description="Classificador de documentos utilizando diversas técnicas de inteligência artificial",
    doc="/api/doc/",
)

logger.info(
    ">>>>>>>>>>>>>>>>>> Acesse a documentação em: http://localhost:5000/api/doc <<<<<<<<<<<<<<<<<<"
)


ns_classifier = api.namespace(name="doc_classifier")
api.add_namespace(ns_classifier, path="doc_classifier")


@ns_classifier.route("/general/cossine")
class General_Classifier_Cossine(Resource):
    def post(self):
        if "file" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return {"error": "Nenhum arquivo enviado"}, 400

        file = request.files["file"]

        if file.filename == "":
            logger.error("Nome de arquivo vazio")
            return {"error": "Nome de arquivo vazio"}, 400

        if not file.filename.lower().endswith(".pdf"):
            logger.error("Formato não suportado. Arquivo precisa ser PDF")
            return {"error": "Formato não suportado. Arquivo precisa ser PDF"}, 400

        logger.info(f"Salvando upload")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        return {"classify_result": general_classifier.classify_doc_cossine(filepath)}


@ns_classifier.route("/general/l2")
class General_Classifier_L2(Resource):
    def post(self):
        if "file" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return {"error": "Nenhum arquivo enviado"}, 400

        file = request.files["file"]

        if file.filename == "":
            logger.error("Nome de arquivo vazio")
            return {"error": "Nome de arquivo vazio"}, 400

        if not file.filename.lower().endswith(".pdf"):
            logger.error("Formato não suportado. Arquivo precisa ser PDF")
            return {"error": "Formato não suportado. Arquivo precisa ser PDF"}, 400

        logger.info(f"Salvando upload")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        return {"classify_result": general_classifier.classify_doc_l2(filepath)}


ns_peti_classifier = api.namespace(name="doc_classifier/peti")
api.add_namespace(ns_peti_classifier, path="doc_classifier/peti")


@ns_peti_classifier.route("/kmeans")
class Kmeans_classifier(Resource):
    def post(self):
        if "file" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return {"error": "Nenhum arquivo enviado"}, 400

        file = request.files["file"]

        if file.filename == "":
            logger.error("Nome de arquivo vazio")
            return {"error": "Nome de arquivo vazio"}, 400

        if not file.filename.lower().endswith(".pdf"):
            logger.error("Formato não suportado. Arquivo precisa ser PDF")
            return {"error": "Formato não suportado. Arquivo precisa ser PDF"}, 400

        logger.info(f"Salvando upload")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        return {"classify_result": kmeans_peti_classifier.classify_peti(filepath)}


@ns_peti_classifier.route("/kmeans_embed")
class Kmeans_Embed_classifier(Resource):
    def post(self):
        if "file" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return {"error": "Nenhum arquivo enviado"}, 400

        file = request.files["file"]

        if file.filename == "":
            logger.error("Nome de arquivo vazio")
            return {"error": "Nome de arquivo vazio"}, 400

        if not file.filename.lower().endswith(".pdf"):
            logger.error("Formato não suportado. Arquivo precisa ser PDF")
            return {"error": "Formato não suportado. Arquivo precisa ser PDF"}, 400

        logger.info(f"Salvando upload")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        return {"classify_result": kmeans_embed_peti_classifier.classify_peti(filepath)}


@ns_peti_classifier.route("/embedding/l2")
class Embed_L2_classifier(Resource):
    def post(self):
        if "file" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return {"error": "Nenhum arquivo enviado"}, 400

        file = request.files["file"]

        if file.filename == "":
            logger.error("Nome de arquivo vazio")
            return {"error": "Nome de arquivo vazio"}, 400

        if not file.filename.lower().endswith(".pdf"):
            logger.error("Formato não suportado. Arquivo precisa ser PDF")
            return {"error": "Formato não suportado. Arquivo precisa ser PDF"}, 400

        logger.info(f"Salvando upload")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        return {"classify_result": embedding_peti_classifier.classify_peti_l2(filepath)}


@ns_peti_classifier.route("/embedding/cossine")
class Embed_Cossine_classifier(Resource):
    def post(self):
        if "file" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return {"error": "Nenhum arquivo enviado"}, 400

        file = request.files["file"]

        if file.filename == "":
            logger.error("Nome de arquivo vazio")
            return {"error": "Nome de arquivo vazio"}, 400

        if not file.filename.lower().endswith(".pdf"):
            logger.error("Formato não suportado. Arquivo precisa ser PDF")
            return {"error": "Formato não suportado. Arquivo precisa ser PDF"}, 400

        logger.info(f"Salvando upload")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        return {
            "classify_result": embedding_peti_classifier.classify_peti_cossine(filepath)
        }


if __name__ == "__main__":
    try:
        os.mkdir(app.config["UPLOAD_FOLDER"])
    except:
        pass

    app.run()
