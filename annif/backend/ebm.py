import os
from typing import Any

import joblib
import numpy as np
from ebm4subjects.ebm_model import EbmModel

from annif.corpus.document import DocumentCorpus
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SuggestionBatch, vector_to_suggestions
from annif.util import atomic_save

from . import backend


class EbmBackend(backend.AnnifBackend):
    name = "ebm"

    EBM_PARAMETERS = {
        "collection_name": str,
        "use_altLabels": bool,
        "duck_db_threads": int,
        "embedding_model_name": str,
        "embedding_dimensions": int,
        "chunk_tokenizer": str,
        "max_chunks": int,
        "max_chunk_size": int,
        "chunking_jobs": int,
        "max_sentences": int,
        "hnsw_index_params": dict[str, Any],
        "max_query_hits": int,
        "query_top_k": int,
        "query_jobs": int,
        "xgb_shrinkage": float,
        "xgb_interaction_depth": int,
        "xgb_subsample": float,
        "xgb_rounds": int,
        "xgb_jobs": int,
        "model_args": dict[str, Any],
        "encode_args_vocab": dict[str, Any],
        "encode_args_documents": dict[str, Any],
    }

    DEFAULT_PARAMETERS = {
        "collection_name": "my_collection",
        "use_altLabels": True,
        "duckdb_threads": 42,
        "embedding_model_name": "jinaai/jina-embeddings-v3",
        "embedding_dimensions": 1024,
        "chunk_tokenizer": "tokenizers/punkt/german.pickle",
        "max_chunks": 100,
        "max_chunk_size": 50,
        "chunking_jobs": 1,
        "max_sentences": 10000,
        "hnsw_index_params": {"M": 32, "ef_construction": 256, "ef_search": 256},
        "max_query_hits": 20,
        "query_top_k": 100,
        "query_jobs": 1,
        "xgb_shrinkage": 0.023,
        "xgb_interaction_depth": 7,
        "xgb_subsample": 0.62,
        "xgb_rounds": 812,
        "xgb_jobs": 1,
        "model_args": {"device": "cuda", "trust_remote_code": True},
        "encode_args_vocab": {
            "batch_size": 300,
            "show_progress_bar": True,
            "task": "retrieval.passage",
        },
        "encode_args_documents": {
            "batch_size": 300,
            "show_progress_bar": True,
            "task": "retrieval.passage",
        },
    }

    DB_FILE = "ebm-duck.db"
    MODEL_FILE = "ebm-model.gz"
    TRAIN_FILE = "ebm-train.gz"

    _model = None

    def initialize(self, parallel: bool = False) -> None:
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)

            self.debug(f"loading model from {path}")
            if os.path.exists(path):
                self._model = EbmModel.load(path)
                self.debug("loaded model")
            else:
                raise NotInitializedException(
                    f"model not found at {path}", backend_id=self.backend_id
                )

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        self.info("starting train")
        self._model = EbmModel(
            db_path=os.path.join(self.datadir, self.DB_FILE),
            collection_name=params["collection_name"],
            use_altLabels=params["use_altLabels"],
            duckdb_threads=params["duckdb_threads"],
            embedding_model_name=params["embedding_model_name"],
            embedding_dimensions=params["embedding_dimensions"],
            chunk_tokenizer=params["chunk_tokenizer"],
            max_chunks=params["max_chunks"],
            max_chunk_size=params["max_chunk_size"],
            chunking_jobs=params["chunking_jobs"],
            max_sentences=params["max_sentences"],
            hnsw_index_params=params["hnsw_index_params"],
            max_query_hits=params["max_query_hits"],
            query_top_k=params["query_top_k"],
            query_jobs=params["query_jobs"],
            xgb_shrinkage=params["xgb_shrinkage"],
            xgb_interaction_depth=params["xgb_interaction_depth"],
            xgb_subsample=params["xgb_subsample"],
            xgb_rounds=params["xgb_rounds"],
            xgb_jobs=params["xgb_jobs"],
            model_args=params["model_args"],
            encode_args_vocab=params["encode_args_vocab"],
            encode_args_documents=params["encode_args_documents"],
        )

        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    f"training backend {self.backend_id} with no documents"
                )

            self.info("creating vector database")
            self._model.create_vector_db(
                vocab_in_path=os.path.join(
                    self.project.vocab.datadir, self.project.vocab.INDEX_FILENAME_TTL
                ),
                force=True,
            )

            self.info("preparing training data")
            doc_ids = []
            texts = []
            label_ids = []
            for doc_id, doc in enumerate(corpus.documents):
                for subject_id in [
                    subject_id for subject_id in getattr(doc, "subject_set")
                ]:
                    doc_ids.append(doc_id)
                    texts.append(getattr(doc, "text"))
                    label_ids.append(self.project.subjects[subject_id].uri)

            train_data = self._model.prepare_train(
                doc_ids=doc_ids,
                label_ids=label_ids,
                texts=texts,
                n_jobs=jobs,
            )

            atomic_save(
                obj=train_data,
                dirname=self.datadir,
                filename=self.TRAIN_FILE,
                method=joblib.dump,
            )

        else:
            self.info("reusing cached training data from previous run")
            if not os.path.exists(self._model.db_path):
                raise NotInitializedException(
                    f"database file {self._model.db_path} not found",
                    backend_id=self.backend_id,
                )
            if not os.path.exists(os.path.join(self.datadir, self.TRAIN_FILE)):
                raise NotInitializedException(
                    f"train data file {self.TRAIN_FILE} not found",
                    backend_id=self.backend_id,
                )

            train_data = joblib.load(os.path.join(self.datadir, self.TRAIN_FILE))

        self.info("training model")
        self._model.train(train_data, jobs)

        self.info("saving model")
        atomic_save(self._model, self.datadir, self.MODEL_FILE)

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        candidates = self._model.generate_candidates_batch(
            texts=texts,
            doc_ids=[i for i in range(len(texts))],
        )

        predictions = self._model.predict(candidates)

        suggestions = []
        for doc_predictions in predictions:
            vector = np.zeros(len(self.project.subjects), dtype=np.float32)
            for row in doc_predictions.iter_rows(named=True):
                position = self.project.subjects._uri_idx.get(row["label_id"], 0)
                vector[position] = row["score"]
            suggestions.append(vector_to_suggestions(vector, int(params["limit"])))

        return SuggestionBatch.from_sequence(
            suggestions,
            self.project.subjects,
            limit=int(params.get("limit")),
        )
