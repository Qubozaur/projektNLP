import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from collections import defaultdict



class NLPPipeline:
    def __init__(self, dialogues_json: str = "data/dialogues.json"):
        with open(dialogues_json, encoding="utf-8") as f:
            data = json.load(f)
        self.df = pd.DataFrame(data["dialogues"])
        self.embeddings: Optional[np.ndarray] = None
        self.topics: Optional[list[int]] = None
        self.topic_model = None
        self.sentiment_results: list[dict] = []
        self.ner_results: list[dict] = []


    def compute_embeddings(self, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        print(f"[Embeddings] Ładowanie modelu: {model_name}")
        model = SentenceTransformer(model_name)
        texts = self.df["text"].tolist()
        print(f"[Embeddings] Kodowanie {len(texts)} wypowiedzi...")
        self.embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"[Embeddings] Shape: {self.embeddings.shape}")
        np.save("data/embeddings.npy", self.embeddings)
        return self.embeddings

    def load_embeddings(self) -> np.ndarray:
        self.embeddings = np.load("data/embeddings.npy")
        return self.embeddings


    def run_bertopic(
        self,
        n_topics: int = 10,
        min_topic_size: int = 3,
        use_mmr: bool = True,
        ) -> pd.DataFrame:
    
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer

        if self.embeddings is None:
            raise RuntimeError("Najpierw wywołaj compute_embeddings()")

        umap_model = UMAP(
            n_neighbors=10,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),    
            min_df=2,
        )

        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            representation_model=None, 
            nr_topics=n_topics,
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True,
        )

        texts = self.df["text"].tolist()
        self.topics, probs = self.topic_model.fit_transform(texts, self.embeddings)
        self.df["topic"] = self.topics
        self.df["topic_prob"] = [max(p) if hasattr(p, '__iter__') else p for p in probs]
        topic_info = self.topic_model.get_topic_info()
        topic_info.to_csv("data/topic_info.csv", index=False)
        topic_words = {}
        for tid in topic_info["Topic"].tolist():
            if tid == -1:
                continue
            words = self.topic_model.get_topic(tid)
            if words:
                topic_words[tid] = [w for w, _ in words[:5]]
        with open("data/topic_words.json", "w") as f:
            json.dump(topic_words, f, indent=2)
        print(f"[BERTopic] Znaleziono {len(topic_info)-1} tematów (bez szumu)")
        print(topic_info.head(15).to_string())
        return topic_info

    def evaluate_topics(self) -> dict:
        if self.topic_model is None or self.embeddings is None:
            raise RuntimeError("Najpierw wywołaj run_bertopic() i compute_embeddings()")
        from sklearn.metrics.pairwise import cosine_similarity
        topic_info = self.topic_model.get_topic_info()
        valid_topics = [t for t in topic_info["Topic"] if t != -1]
        all_top_words = []
        for tid in valid_topics:
            words = self.topic_model.get_topic(tid) or []
            all_top_words.extend([w for w, _ in words[:10]])
        diversity = len(set(all_top_words)) / len(all_top_words) if all_top_words else 0.0
        coherence_scores = []
        for tid in valid_topics:
            mask = np.array(self.topics) == tid
            if mask.sum() < 2:
                continue
            topic_embs = self.embeddings[mask]
            sim_matrix = cosine_similarity(topic_embs)
            upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            coherence_scores.append(upper.mean())
        mean_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        assigned = sum(1 for t in self.topics if t != -1)
        coverage = assigned / len(self.topics)
        metrics = {
            "topic_diversity": round(diversity, 4),
            "mean_intra_topic_coherence": round(mean_coherence, 4),
            "topic_coverage": round(coverage, 4),
            "n_topics": len(valid_topics),
            "n_noise_docs": sum(1 for t in self.topics if t == -1),
        }
        with open("data/topic_evaluation.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\n[Ewaluacja tematów]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return metrics

    def evaluate_topics_coherence(self) -> dict:
        import re
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        if self.topic_model is None or self.topics is None:
            raise RuntimeError("Najpierw wywołaj run_bertopic()")

        tokenized = []
        for text in self.df["text"].tolist():
            tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
            tokenized.append([t for t in tokens if t not in ENGLISH_STOP_WORDS])

        topic_info = self.topic_model.get_topic_info()
        valid_topics = [t for t in topic_info["Topic"] if t != -1]
        topics_words = []
        for tid in valid_topics:
            words = self.topic_model.get_topic(tid) or []
            topics_words.append([w for w, _ in words[:10]])

        dictionary = Dictionary(tokenized)
        cm = CoherenceModel(
            topics=topics_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        per_topic = cm.get_coherence_per_topic()
        mean_cv = cm.get_coherence()

        result = {
            "mean_coherence_cv": round(mean_cv, 4),
            "per_topic_coherence_cv": {
                str(tid): round(score, 4)
                for tid, score in zip(valid_topics, per_topic)
            },
        }
        with open("data/topic_coherence_cv.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Coherence Cv] Średnia: {mean_cv:.4f}")
        for tid, score in zip(valid_topics, per_topic):
            print(f"  T{tid}: {score:.4f}")
        return result

    def character_arc_analysis(self) -> pd.DataFrame:
        if "sentiment" not in self.df.columns:
            raise RuntimeError("Najpierw wywołaj run_sentiment()")

        chapter_order = ["2a","2b","2c","2d","2e","2f","2g","2h","2i","2j","2k","2l","2m","2n"]
        sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

        df2 = self.df.copy()
        df2["sentiment_val"] = df2["sentiment"].map(sentiment_map).fillna(0.0)

        main_chars = df2["character"].value_counts().head(10).index.tolist()

        records = []
        for char in main_chars:
            char_df = df2[df2["character"] == char]
            for ch in chapter_order:
                ch_df = char_df[char_df["chapter_id"] == ch]
                if len(ch_df) == 0:
                    continue
                records.append({
                    "character": char,
                    "chapter_id": ch,
                    "mean_sentiment": round(ch_df["sentiment_val"].mean(), 3),
                    "n_lines": len(ch_df),
                    "pos_ratio": round((ch_df["sentiment"] == "positive").mean(), 3),
                    "neg_ratio": round((ch_df["sentiment"] == "negative").mean(), 3),
                })

        arc_df = pd.DataFrame(records)
        arc_df.to_csv("data/character_arcs.csv", index=False)
        print(f"\n[Character Arcs] {len(arc_df)} wpisów zapisanych do data/character_arcs.csv")
        return arc_df

    def speech_style_analysis(self) -> pd.DataFrame:
        import re

        stop_words = {
            "i", "a", "the", "and", "to", "of", "in", "you", "it", "is", "that",
            "this", "was", "he", "she", "they", "we", "be", "are", "have", "has",
            "had", "do", "does", "did", "not", "but", "on", "at", "by", "for",
            "with", "his", "her", "my", "your", "our", "what", "all", "there",
            "from", "will", "an", "can", "if", "as", "me", "him", "them", "us",
            "just", "so", "up", "out", "no", "get", "oh", "well", "yes",
        }

        records = []
        for char, group in self.df.groupby("character"):
            if len(group) < 3:
                continue
            texts = group["text"].tolist()

            avg_words = np.mean([len(t.split()) for t in texts])
            question_ratio = sum(1 for t in texts if "?" in t) / len(texts)
            exclamation_ratio = sum(1 for t in texts if "!" in t) / len(texts)
            ellipsis_ratio = sum(1 for t in texts if "..." in t) / len(texts)

            all_words = []
            for t in texts:
                tokens = re.findall(r'\b[a-z]{2,}\b', t.lower())
                all_words.extend([w for w in tokens if w not in stop_words])
            vocab_richness = len(set(all_words)) / len(all_words) if all_words else 0.0

            records.append({
                "character": char,
                "n_lines": len(group),
                "avg_words_per_line": round(avg_words, 2),
                "question_ratio": round(question_ratio, 3),
                "exclamation_ratio": round(exclamation_ratio, 3),
                "ellipsis_ratio": round(ellipsis_ratio, 3),
                "vocabulary_richness": round(vocab_richness, 3),
            })

        style_df = pd.DataFrame(records).sort_values("n_lines", ascending=False)
        style_df.to_csv("data/speech_style.csv", index=False)
        print(f"\n[Speech Style] Analiza {len(style_df)} postaci zapisana do data/speech_style.csv")
        return style_df

    def run_sentiment(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        batch_size: int = 16,
    ) -> pd.DataFrame:
        from transformers import pipeline
        print(f"[Sentiment] Ładowanie modelu: {model_name}")
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=512,
            device=-1, 
        )
        texts = self.df["text"].tolist()
        print(f"[Sentiment] Analiza {len(texts)} wypowiedzi...")
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            preds = sentiment_pipe(batch)
            results.extend(preds)
        self.df["sentiment_label"] = [r["label"] for r in results]
        self.df["sentiment_score"] = [r["score"] for r in results]
        label_map = {
            "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
            "NEGATIVE": "negative", "NEUTRAL": "neutral", "POSITIVE": "positive",
            "negative": "negative", "neutral": "neutral", "positive": "positive",
        }
        self.df["sentiment"] = self.df["sentiment_label"].map(
            lambda x: label_map.get(x.upper(), x.lower())
        )
        self.df[["chapter_id", "character", "text", "sentiment", "sentiment_score"]].to_csv(
            "data/sentiment_results.csv", index=False
        )
        print("[Sentiment] Gotowe. Wyniki w data/sentiment_results.csv")
        return self.df

    def run_ner(self, model: str = "en_core_web_sm") -> list[dict]:
        import spacy
        print(f"[NER] Ładowanie modelu spaCy: {model}")
        nlp = spacy.load(model)
        results = []
        for _, row in self.df.iterrows():
            doc = nlp(row["text"])
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ]
            results.append({
                "chapter_id": row["chapter_id"],
                "character": row["character"],
                "line": row["text"],
                "entities": entities,
            })
        entity_freq: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        for r in results:
            for ent in r["entities"]:
                entity_freq[ent["label"]][ent["text"]] += 1
        self.ner_results = results
        with open("data/ner_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n[NER] Najczęstsze encje:")
        for label, counts in entity_freq.items():
            top = sorted(counts.items(), key=lambda x: -x[1])[:5]
            print(f"  {label}: {top}")
        return results


    def character_similarity_matrix(self) -> pd.DataFrame:
        from sklearn.metrics.pairwise import cosine_similarity
        if self.embeddings is None:
            raise RuntimeError("Najpierw wywołaj compute_embeddings()")
        character_centroids: dict[str, np.ndarray] = {}
        for char in self.df["character"].unique():
            mask = self.df["character"] == char
            if mask.sum() < 2:
                continue
            char_embs = self.embeddings[mask.values]
            character_centroids[char] = char_embs.mean(axis=0)
        chars = list(character_centroids.keys())
        matrix = np.array([character_centroids[c] for c in chars])
        sim = cosine_similarity(matrix)
        sim_df = pd.DataFrame(sim, index=chars, columns=chars)
        sim_df.to_csv("data/character_similarity.csv")
        print("\n[Similarity] Macierz podobieństwa postaci zapisana.")
        return sim_df


    def topic_timeline(self) -> pd.DataFrame:
        if "topic" not in self.df.columns:
            raise RuntimeError("Najpierw wywołaj run_bertopic()")
        timeline = (
            self.df[self.df["topic"] != -1]
            .groupby(["chapter_id", "topic"])
            .size()
            .reset_index(name="count")
        )
        timeline.to_csv("data/topic_timeline.csv", index=False)
        return timeline

    def run_all(self, recompute_embeddings: bool = False):
        Path("data").mkdir(exist_ok=True)
        if recompute_embeddings or not Path("data/embeddings.npy").exists():
            self.compute_embeddings()
        else:
            print("[Pipeline] Wczytywanie istniejących embeddingów...")
            self.load_embeddings()
        topic_info = self.run_bertopic(n_topics=10)
        eval_metrics = self.evaluate_topics()
        coherence_cv = self.evaluate_topics_coherence()
        self.run_sentiment()
        self.run_ner()
        arc_df = self.character_arc_analysis()
        style_df = self.speech_style_analysis()
        sim_df = self.character_similarity_matrix()
        timeline = self.topic_timeline()
        self.df.to_csv("data/full_analysis.csv", index=False)
        print("\n[Pipeline] Kompletna analiza zakończona. Dane w katalogu data/")
        return {
            "df": self.df,
            "topic_info": topic_info,
            "eval_metrics": eval_metrics,
            "coherence_cv": coherence_cv,
            "similarity": sim_df,
            "timeline": timeline,
            "character_arcs": arc_df,
            "speech_style": style_df,
        }


if __name__ == "__main__":
    pipeline = NLPPipeline("data/dialogues.json")
    results = pipeline.run_all(recompute_embeddings=True)