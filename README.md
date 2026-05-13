# projektNLP — analiza dialogów z Half-Life 2

Projekt na zajęcia z NLP. Bierzemy oficjalny skrypt gry **Half-Life 2** (PDF),
wyciągamy z niego dialogi i przepuszczamy je przez kilka klasycznych zadań NLP:

- **topic modeling** (BERTopic + UMAP + HDBSCAN) — wykrywanie motywów dialogów
- **sentiment analysis** (RoBERTa z HuggingFace) — wydźwięk wypowiedzi
- **NER** (spaCy) — wykrywanie nazwanych encji
- **stylometria** postaci (długość wypowiedzi, pytajniki, bogactwo słownika)
- **embeddingi** i podobieństwo postaci (sentence-transformers + cosine sim)
- **łuki emocjonalne** głównych postaci na przestrzeni rozdziałów
- **dashboard** w Dash + Plotly do oglądania wyników
- **generator wypowiedzi** w stylu konkretnej postaci (OpenAI API)

## Struktura repo

```
projektNLP/
├── main.py                  # placeholder, nieużywany
├── data_parser.py           # parser PDF -> JSON z dialogami
├── nlp_pipeline.py          # pipeline NLP + ewaluacja tematów
├── character_generator.py   # generator wypowiedzi (OpenAI)
├── dashboard.py             # aplikacja Dash z wykresami
├── pyproject.toml           # zależności (uv)
├── requirements.txt         # zależności (pip)
├── .env                     # OPENAI_API_KEY (nie commitowane)
├── data/
│   ├── half_life_2_script.pdf   # surowe dane wejściowe
│   ├── dialogues.json           # wynik parsera
│   ├── embeddings.npy           # cache embeddingów
│   ├── full_analysis.csv        # wszystkie dialogi + tematy + sentyment
│   ├── topic_info.csv           # info o tematach
│   ├── topic_words.json         # top 5 słów na temat
│   ├── topic_evaluation.json    # metryki ewaluacji tematów
│   ├── topic_coherence_cv.json  # coherence Cv (gensim)
│   ├── topic_timeline.csv       # rozkład tematów po rozdziałach
│   ├── topic_motifs.json        # cache opisów motywów (LLM)
│   ├── sentiment_results.csv    # wyniki sentymentu
│   ├── ner_results.json         # wyniki NER
│   ├── character_arcs.csv       # łuki emocjonalne postaci
│   ├── character_similarity.csv # macierz podobieństwa postaci
│   └── speech_style.csv         # stylometria postaci
└── assets/                  # statyczne pliki dla dashboardu
```

## Dane

Źródło: oficjalny PDF ze skryptem Half-Life 2 dołączony w `data/half_life_2_script.pdf`.
Plik jest w repo — nie trzeba go nigdzie pobierać.

Jeśli plik się zgubi, pobrać można go np. ze strony combineoverwiki / valve archive.
**Ostatnia data testowania parsera na tym PDF: 2026-05-13.**

## Wymagania

- Python **3.12+**
- (opcjonalnie) klucz OpenAI w pliku `.env` — tylko jeśli chcemy używać generatora
  wypowiedzi albo automatycznych opisów motywów w dashboardzie:

  ```
  OPENAI_API_KEY=sk-...
  ```

## Instalacja

**Wariant A — `uv` (zalecane):**

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

**Wariant B — czysty `pip` / venv:**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install bertopic sentence-transformers spacy umap-learn hdbscan ^
            plotly dash dash-bootstrap-components transformers torch ^
            pdfplumber pandas numpy scikit-learn openai python-dotenv ^
            gensim
python -m spacy download en_core_web_sm
```

Pierwsze uruchomienie pobierze modele (sentence-transformers + RoBERTa
sentiment), więc będzie chwilę trwało.

## Jak uruchomić

### 1. Parsowanie PDF → JSON z dialogami

```bash
uv run python data_parser.py
```

Wynik: `data/dialogues.json` (lista wypowiedzi z polem postać / rozdział / tekst).
W repo jest już gotowy plik, więc ten krok można pominąć.

### 2. Pipeline NLP + ewaluacja

```bash
uv run python nlp_pipeline.py
```

To uruchamia całą metodę `Analiza.wszystko(...)`:
robi embeddingi, znajduje tematy, ocenia je, liczy sentyment, NER, stylometrię,
podobieństwo postaci i łuki emocjonalne. Wszystko zapisuje do `data/`.

Pierwsze uruchomienie liczy embeddingi od zera.
Kolejne uruchomienia wczytują `data/embeddings.npy`, chyba że ustawimy
`recompute_embeddings=True`.

### 3. Dashboard

```bash
uv run python dashboard.py
```

Następnie otwórz w przeglądarce: <http://127.0.0.1:8050>

W dashboardzie są wykresy tematów, sentymentu, podobieństwa postaci,
łuków emocjonalnych, stylometrii oraz panel do generowania wypowiedzi.

### 4. Generator wypowiedzi (opcjonalnie)

Wymaga `OPENAI_API_KEY` w `.env`.

```bash
uv run python character_generator.py
```

Wypisze w terminalu po jednej wygenerowanej wypowiedzi dla każdej z głównych
postaci (G-Man, Alyx, Breen, Barney, Father Gregori, Vortigaunt) oraz krótką
przykładową scenę.

## Ewaluacja

Pipeline liczy następujące metryki dla wykrytych tematów:

- **topic_diversity** — udział unikalnych słów w top słowach wszystkich tematów
- **mean_intra_topic_coherence** — średnia cosine similarity wewnątrz klastra
- **topic_coverage** — % dokumentów przypisanych do jakiegoś tematu (nie do szumu)
- **mean_coherence_cv** — coherence Cv z gensim (klasyczna metryka dla LDA/BERTopic)
- liczba tematów oraz liczba dokumentów potraktowanych jako szum (`-1`)

Wyniki lądują w `data/topic_evaluation.json` i `data/topic_coherence_cv.json`.

Dla sentymentu nie ma zbioru testowego z etykietami (skrypt gry to nie benchmark),
więc model RoBERTa traktujemy jako gotowy i jego wyniki oglądamy jakościowo
w dashboardzie (rozkład per postać, łuki emocjonalne).

Dla generatora wypowiedzi jest funkcja `Generator.porownaj_styl(...)`, która
porównuje stylometrię oryginalnych linii postaci i linii wygenerowanych
(różnica średniej długości, użycia pytajników/wykrzykników/wielokropków,
pokrycie top-słów).

## Uwagi

- Wszystkie ciężkie pliki (`embeddings.npy`, dane wynikowe) są w `data/`
  i komitujemy je razem z kodem, żeby projekt dał się odpalić bez liczenia
  od zera.
