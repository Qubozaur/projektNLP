import json
import numpy as np
import pandas as pd
from pathlib import Path


class Analiza:
    def __init__(self, plik="data/dialogues.json"):
        with open(plik, encoding="utf-8") as f:
            dane = json.load(f)
        self.df = pd.DataFrame(dane["dialogues"])
        self.embeddings = None
        self.topics = None
        self.topic_model = None
        self.ner_results = []

    def zrob_embeddingi(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print("Laduje model embeddingow:", model_name)
        model = SentenceTransformer(model_name)
        teksty = self.df["text"].tolist()
        print("Koduje", len(teksty), "wypowiedzi...")
        self.embeddings = model.encode(
            teksty,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print("Wymiary embeddingow:", self.embeddings.shape)
        np.save("data/embeddings.npy", self.embeddings)
        return self.embeddings

    def wczytaj_embeddingi(self):
        self.embeddings = np.load("data/embeddings.npy")
        return self.embeddings

    def tematy(self, n_topics=10, min_topic_size=3, use_mmr=True):
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer

        if self.embeddings is None:
            raise RuntimeError("Najpierw zrob embeddingi")

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

        vect = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )

        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vect,
            representation_model=None,
            nr_topics=n_topics,
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True,
        )

        teksty = self.df["text"].tolist()
        self.topics, probs = self.topic_model.fit_transform(teksty, self.embeddings)
        self.df["topic"] = self.topics

        proby = []
        for p in probs:
            if hasattr(p, '__iter__'):
                proby.append(max(p))
            else:
                proby.append(p)
        self.df["topic_prob"] = proby

        info = self.topic_model.get_topic_info()
        info.to_csv("data/topic_info.csv", index=False)

        slowa = {}
        for tid in info["Topic"].tolist():
            if tid == -1:
                continue
            w = self.topic_model.get_topic(tid)
            if w:
                top5 = []
                for slowo, _ in w[:5]:
                    top5.append(slowo)
                slowa[tid] = top5
        with open("data/topic_words.json", "w") as f:
            json.dump(slowa, f, indent=2)

        print("Znaleziono", len(info) - 1, "tematow (bez szumu)")
        print(info.head(15).to_string())
        return info

    def ocena(self):
        if self.topic_model is None or self.embeddings is None:
            raise RuntimeError("Najpierw zrob tematy i embeddingi")
        from sklearn.metrics.pairwise import cosine_similarity

        info = self.topic_model.get_topic_info()
        ok_tematy = []
        for t in info["Topic"]:
            if t != -1:
                ok_tematy.append(t)

        wszystkie_slowa = []
        for tid in ok_tematy:
            w = self.topic_model.get_topic(tid) or []
            for slowo, _ in w[:10]:
                wszystkie_slowa.append(slowo)
        if wszystkie_slowa:
            diversity = len(set(wszystkie_slowa)) / len(wszystkie_slowa)
        else:
            diversity = 0.0

        wyniki = []
        for tid in ok_tematy:
            maska = np.array(self.topics) == tid
            if maska.sum() < 2:
                continue
            embs = self.embeddings[maska]
            sim = cosine_similarity(embs)
            gora = sim[np.triu_indices_from(sim, k=1)]
            wyniki.append(gora.mean())
        if wyniki:
            coherence = float(np.mean(wyniki))
        else:
            coherence = 0.0

        przypisane = 0
        for t in self.topics:
            if t != -1:
                przypisane += 1
        coverage = przypisane / len(self.topics)

        szum = 0
        for t in self.topics:
            if t == -1:
                szum += 1

        metryki = {
            "topic_diversity": round(diversity, 4),
            "mean_intra_topic_coherence": round(coherence, 4),
            "topic_coverage": round(coverage, 4),
            "n_topics": len(ok_tematy),
            "n_noise_docs": szum,
        }
        with open("data/topic_evaluation.json", "w") as f:
            json.dump(metryki, f, indent=2)
        print("\nOcena tematow:")
        for k in metryki:
            print(" ", k, ":", metryki[k])
        return metryki

    def coherence(self):
        import re
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        if self.topic_model is None or self.topics is None:
            raise RuntimeError("Najpierw zrob tematy")

        tokeny = []
        for tekst in self.df["text"].tolist():
            t = re.findall(r'\b[a-z]{2,}\b', tekst.lower())
            t2 = []
            for w in t:
                if w not in ENGLISH_STOP_WORDS:
                    t2.append(w)
            tokeny.append(t2)

        info = self.topic_model.get_topic_info()
        ok_tematy = []
        for t in info["Topic"]:
            if t != -1:
                ok_tematy.append(t)
        slowa_tematow = []
        for tid in ok_tematy:
            w = self.topic_model.get_topic(tid) or []
            top10 = []
            for slowo, _ in w[:10]:
                top10.append(slowo)
            slowa_tematow.append(top10)

        slownik = Dictionary(tokeny)
        cm = CoherenceModel(
            topics=slowa_tematow,
            texts=tokeny,
            dictionary=slownik,
            coherence="c_v",
        )
        per_temat = cm.get_coherence_per_topic()
        srednia = cm.get_coherence()

        per_temat_dict = {}
        for tid, score in zip(ok_tematy, per_temat):
            per_temat_dict[str(tid)] = round(score, 4)

        wynik = {
            "mean_coherence_cv": round(srednia, 4),
            "per_topic_coherence_cv": per_temat_dict,
        }
        with open("data/topic_coherence_cv.json", "w") as f:
            json.dump(wynik, f, indent=2)
        print("\nCoherence Cv - srednia:", round(srednia, 4))
        for tid, score in zip(ok_tematy, per_temat):
            print("  T" + str(tid) + ":", round(score, 4))
        return wynik

    def arc_postaci(self):
        if "sentiment" not in self.df.columns:
            raise RuntimeError("Najpierw zrob sentyment")

        rozdzialy = ["2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h", "2i", "2j", "2k", "2l", "2m", "2n"]
        mapa = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        df2 = self.df.copy()
        df2["sentiment_val"] = df2["sentiment"].map(mapa).fillna(0.0)
        glowne = df2["character"].value_counts().head(10).index.tolist()

        wpisy = []
        for postac in glowne:
            df_p = df2[df2["character"] == postac]
            for r in rozdzialy:
                df_r = df_p[df_p["chapter_id"] == r]
                if len(df_r) == 0:
                    continue
                wpisy.append({
                    "character": postac,
                    "chapter_id": r,
                    "mean_sentiment": round(df_r["sentiment_val"].mean(), 3),
                    "n_lines": len(df_r),
                    "pos_ratio": round((df_r["sentiment"] == "positive").mean(), 3),
                    "neg_ratio": round((df_r["sentiment"] == "negative").mean(), 3),
                })

        arc = pd.DataFrame(wpisy)
        arc.to_csv("data/character_arcs.csv", index=False)
        print("\nArc postaci -", len(arc), "wpisow zapisano do data/character_arcs.csv")
        return arc

    def styl(self):
        import re
        stopwordy = {
            "i", "a", "the", "and", "to", "of", "in", "you", "it", "is", "that",
            "this", "was", "he", "she", "they", "we", "be", "are", "have", "has",
            "had", "do", "does", "did", "not", "but", "on", "at", "by", "for",
            "with", "his", "her", "my", "your", "our", "what", "all", "there",
            "from", "will", "an", "can", "if", "as", "me", "him", "them", "us",
            "just", "so", "up", "out", "no", "get", "oh", "well", "yes",
        }

        wpisy = []
        for postac, grupa in self.df.groupby("character"):
            if len(grupa) < 3:
                continue
            teksty = grupa["text"].tolist()

            dlug = []
            for t in teksty:
                dlug.append(len(t.split()))
            srednia_slow = np.mean(dlug)

            ile_p = 0
            ile_w = 0
            ile_e = 0
            for t in teksty:
                if "?" in t:
                    ile_p += 1
                if "!" in t:
                    ile_w += 1
                if "..." in t:
                    ile_e += 1
            ratio_p = ile_p / len(teksty)
            ratio_w = ile_w / len(teksty)
            ratio_e = ile_e / len(teksty)

            wszystkie = []
            for t in teksty:
                tokeny = re.findall(r'\b[a-z]{2,}\b', t.lower())
                for w in tokeny:
                    if w not in stopwordy:
                        wszystkie.append(w)
            if wszystkie:
                bogactwo = len(set(wszystkie)) / len(wszystkie)
            else:
                bogactwo = 0.0

            wpisy.append({
                "character": postac,
                "n_lines": len(grupa),
                "avg_words_per_line": round(srednia_slow, 2),
                "question_ratio": round(ratio_p, 3),
                "exclamation_ratio": round(ratio_w, 3),
                "ellipsis_ratio": round(ratio_e, 3),
                "vocabulary_richness": round(bogactwo, 3),
            })

        styl_df = pd.DataFrame(wpisy).sort_values("n_lines", ascending=False)
        styl_df.to_csv("data/speech_style.csv", index=False)
        print("\nStyl -", len(styl_df), "postaci zapisano do data/speech_style.csv")
        return styl_df

    def sentyment(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", batch_size=16):
        from transformers import pipeline
        print("Laduje model sentymentu:", model_name)
        sent = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=512,
            device=-1,
        )
        teksty = self.df["text"].tolist()
        print("Analizuje", len(teksty), "wypowiedzi...")

        wyniki = []
        for i in range(0, len(teksty), batch_size):
            batch = teksty[i:i + batch_size]
            pred = sent(batch)
            wyniki.extend(pred)

        labelki = []
        score = []
        for r in wyniki:
            labelki.append(r["label"])
            score.append(r["score"])
        self.df["sentiment_label"] = labelki
        self.df["sentiment_score"] = score

        mapa = {
            "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
            "NEGATIVE": "negative", "NEUTRAL": "neutral", "POSITIVE": "positive",
            "negative": "negative", "neutral": "neutral", "positive": "positive",
        }
        out = []
        for x in self.df["sentiment_label"]:
            klucz = x.upper()
            if klucz in mapa:
                out.append(mapa[klucz])
            else:
                out.append(x.lower())
        self.df["sentiment"] = out

        self.df[["chapter_id", "character", "text", "sentiment", "sentiment_score"]].to_csv(
            "data/sentiment_results.csv", index=False
        )
        print("Sentyment gotowy - dane w data/sentiment_results.csv")
        return self.df

    def ner(self, model="en_core_web_sm"):
        import spacy
        print("Laduje spaCy:", model)
        nlp = spacy.load(model)
        wyniki = []
        for _, wiersz in self.df.iterrows():
            doc = nlp(wiersz["text"])
            encje = []
            for ent in doc.ents:
                encje.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
            wyniki.append({
                "chapter_id": wiersz["chapter_id"],
                "character": wiersz["character"],
                "line": wiersz["text"],
                "entities": encje,
            })

        czestosci = {}
        for r in wyniki:
            for e in r["entities"]:
                label = e["label"]
                tekst = e["text"]
                if label not in czestosci:
                    czestosci[label] = {}
                if tekst not in czestosci[label]:
                    czestosci[label][tekst] = 0
                czestosci[label][tekst] += 1

        self.ner_results = wyniki
        with open("data/ner_results.json", "w", encoding="utf-8") as f:
            json.dump(wyniki, f, ensure_ascii=False, indent=2)

        print("\nNajczestsze encje:")
        for label in czestosci:
            top = sorted(czestosci[label].items(), key=lambda x: -x[1])[:5]
            print(" ", label, ":", top)
        return wyniki

    def podobienstwo(self):
        from sklearn.metrics.pairwise import cosine_similarity
        if self.embeddings is None:
            raise RuntimeError("Najpierw zrob embeddingi")

        srednie = {}
        for postac in self.df["character"].unique():
            maska = self.df["character"] == postac
            if maska.sum() < 2:
                continue
            embs = self.embeddings[maska.values]
            srednie[postac] = embs.mean(axis=0)

        postacie = list(srednie.keys())
        macierz = []
        for c in postacie:
            macierz.append(srednie[c])
        macierz = np.array(macierz)
        sim = cosine_similarity(macierz)

        sim_df = pd.DataFrame(sim, index=postacie, columns=postacie)
        sim_df.to_csv("data/character_similarity.csv")
        print("\nMacierz podobienstwa postaci zapisana.")
        return sim_df

    def timeline_tematow(self):
        if "topic" not in self.df.columns:
            raise RuntimeError("Najpierw zrob tematy")
        tl = (
            self.df[self.df["topic"] != -1]
            .groupby(["chapter_id", "topic"])
            .size()
            .reset_index(name="count")
        )
        tl.to_csv("data/topic_timeline.csv", index=False)
        return tl

    def wszystko(self, recompute_embeddings=False):
        Path("data").mkdir(exist_ok=True)
        if recompute_embeddings or not Path("data/embeddings.npy").exists():
            self.zrob_embeddingi()
        else:
            print("Wczytuje istniejace embeddingi...")
            self.wczytaj_embeddingi()

        info = self.tematy(n_topics=10)
        metryki = self.ocena()
        coh = self.coherence()
        self.sentyment()
        self.ner()
        arc = self.arc_postaci()
        st = self.styl()
        sim = self.podobienstwo()
        tl = self.timeline_tematow()

        self.df.to_csv("data/full_analysis.csv", index=False)
        print("\nGotowe. Dane w katalogu data/")
        return {
            "df": self.df,
            "topic_info": info,
            "eval_metrics": metryki,
            "coherence_cv": coh,
            "similarity": sim,
            "timeline": tl,
            "character_arcs": arc,
            "speech_style": st,
        }


if __name__ == "__main__":
    a = Analiza("data/dialogues.json")
    wyniki = a.wszystko(recompute_embeddings=True)
