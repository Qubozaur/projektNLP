import json
import re
from openai import OpenAI
from openai import OpenAIError
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# profile postaci ich opis i przyklady linii do promptu
CHARACTER_PROFILES = {
    "G-Man": {
        "description": "Mysterious interdimensional bureaucrat. Speaks in an ominous, slow, deliberately cryptic manner. Uses formal, archaic phrasing. Treats time and reality as commodities. Refers to himself in formal third-person occasionally. Never reveals his true intentions.",
        "style_notes": "Ellipses mid-sentence... Unusual pauses. Metaphors about time and employment. Formal but unsettling.",
        "example_lines": [
            "Rise and shine, Mr. Freeman, rise and shine.",
            "The right man in the wrong place can make all the difference in the world.",
            "I do apologize for what must seem to you an arbitrary imposition.",
        ],
    },
    "Alyx": {
        "description": "Young, witty, capable resistance fighter. Warm and sarcastic in turns. Deeply loyal to her father and Gordon. Speaks casually and directly, often with humor under pressure.",
        "style_notes": "Contractions. Light sarcasm. Emotional authenticity. Short punchy sentences. Occasional 'uh-oh'.",
        "example_lines": [
            "Man of few words aren't you?",
            "I wasn't entirely sure you were ever going to get around to that.",
            "Yeah! You did it! Come on Gordon we've got to get out of here.",
        ],
    },
    "Dr. Breen": {
        "description": "Former Black Mesa administrator turned collaborator with the Combine. Speaks in grandiose, politically manipulative language. Condescending but charming. Sees himself as humanity's savior.",
        "style_notes": "Long rhetorical sentences. Political doublespeak. References to history and evolution. Appeals to reason while being fundamentally dishonest.",
        "example_lines": [
            "Welcome, welcome to City 17.",
            "Let me remind all citizens of the dangers of magical thinking.",
            "You have my gratitude, Dr. Freeman.",
        ],
    },
    "Barney": {
        "description": "Gordon's old friend from Black Mesa, now undercover in Civil Protection. Casual, wisecracking, salt-of-the-earth. Loyal and brave but self-deprecating.",
        "style_notes": "Colloquial. Humor as coping mechanism. Blue-collar directness. Short sentences.",
        "example_lines": [
            "About that beer I owed ya.",
            "I still have nightmares about that cat.",
            "Good luck out there buddy, you're gonna need it.",
        ],
    },
    "Dr. Kleiner": {
        "description": "Eccentric but brilliant physicist. Absent-minded, enthusiastic, endearingly formal. Sees everything through the lens of science. Devoted to his work and his headcrab Lamarr.",
        "style_notes": "Scientific terminology mixed with flustered exclamations. 'Great Scott!' energy. Long digressions.",
        "example_lines": [
            "Great Scott! Gordon Freeman!",
            "The massless field flux should self-limit.",
            "Oh fie! It will be another week before I can coax her out of there!",
        ],
    },
    "Father Gregori": {
        "description": "Half-mad priest surviving alone in zombie-infested Ravenholm. Quotes scripture, laughs maniacally, deeply committed to his 'flock' of zombies. Righteous and terrifying.",
        "style_notes": "Biblical cadence. Dark humor. Third-person references to 'brother'. Non-sequiturs of salvation.",
        "example_lines": [
            "May they become like dust before the wind.",
            "In Ravenholm, you do well to be vigilant.",
            "A shepherd must tend to his flock... especially when they have grown unruly.",
        ],
    },
    "Eli": {
        "description": "Elder scientist and resistance leader. Warm, fatherly, principled. Mentor figure who has seen tremendous loss. Speaks with weight and authority.",
        "style_notes": "Measured. Emotional depth. Fatherly warmth. Occasional wry humor.",
        "example_lines": [
            "I never thought it would take you this long to get back to me.",
            "God damn you Breen! You let her go!",
            "MIT graduates are few and far between these days.",
        ],
    },
    "Vortigaunt": {
        "description": "Alien slave-race now freed and allied with humanity. Speaks with an alien cadence, referring to Gordon as 'the Freeman'. Philosophical, ancient, spiritual.",
        "style_notes": "Third-person references. Alien idiom. Mystical tone. 'The Freeman'. 'We serve the same mystery.'",
        "example_lines": [
            "This is the Freeman. The reckoning of the Combine has come.",
            "We serve the same mystery.",
            "The Freeman will accept this weapon, or suffer greatly on the road ahead.",
        ],
    },
}


def korpus_postaci(df):
    """Zbiera wszystkie linie dialogowe dla kazdej postaci.

    Bierze tylko postacie, ktore maja zdefiniowany profil w
    CHARACTER_PROFILES (czyli te, dla ktorych umiemy generowac).

    Args:
        df: DataFrame z kolumnami "character" i "text".

    Returns:
        dict[str, list[str]]: {nazwa_postaci: [linie tej postaci]}.
    """
    korpus = {}
    for _, wiersz in df.iterrows():
        postac = wiersz["character"]
        if postac in CHARACTER_PROFILES:
            if postac not in korpus:
                korpus[postac] = []
            korpus[postac].append(wiersz["text"])
    return korpus


def statystyki(linie):
    """Liczy stylometryczne statystyki dla listy linii dialogowych.

    Sluzy do dwoch rzeczy:
    1. Karmienia promptu LLM "stylometric profile" w _prompt().
    2. Porownywania oryginalu z wygenerowanymi liniami w porownaj_styl().

    Args:
        linie: Lista stringow (wypowiedzi postaci).

    Returns:
        dict: avg_words_per_line, question_ratio, exclamation_ratio,
            ellipsis_ratio, top_words (top 10), total_lines.
            Pusty dict jesli linie sa puste.
    """
    if not linie:
        return {}

    suma = 0
    for l in linie:
        suma += len(l.split())
    srednia = suma / len(linie)

    p = 0
    w = 0
    e = 0
    for l in linie:
        if "?" in l:
            p += 1
        if "!" in l:
            w += 1
        if "..." in l:
            e += 1
    ratio_p = p / len(linie)
    ratio_w = w / len(linie)
    ratio_e = e / len(linie)

    stop = {"the", "a", "an", "i", "you", "we", "he", "she", "it",
            "to", "of", "and", "in", "is", "have", "that", "this",
            "are", "was", "be", "not", "with", "for", "my", "your"}
    freq = {}
    for l in linie:
        for slowo in re.findall(r"\b\w+\b", l.lower()):
            if slowo not in stop and len(slowo) > 3:
                if slowo not in freq:
                    freq[slowo] = 0
                freq[slowo] += 1

    top = sorted(freq.items(), key=lambda x: -x[1])[:10]
    top_slowa = []
    for s, _ in top:
        top_slowa.append(s)

    return {
        "avg_words_per_line": round(srednia, 1),
        "question_ratio": round(ratio_p, 2),
        "exclamation_ratio": round(ratio_w, 2),
        "ellipsis_ratio": round(ratio_e, 2),
        "top_words": top_slowa,
        "total_lines": len(linie),
    }


class Generator:
    """Generuje wypowiedzi w stylu konkretnej postaci z Half-Life 2.

    Pod spodem leci OpenAI Chat Completions. Prompt budowany jest z trzech
    rzeczy:
    1. Recznie napisany opis postaci z CHARACTER_PROFILES.
    2. Przykladowe linie ze skryptu (jesli mamy df).
    3. Stylometryczne statystyki (jesli mamy df) - srednia dlugosc, udzial
       pytajnikow itp.

    Bez df nadal dziala, ale slabiej. Bez df ma tylko sam profil + przykladowe linie
    zaszyte recznie w kodzie.
    """

    def __init__(self, api_key=None, model="gpt-4o-mini", df=None):
        """Inicjalizuje generator.

        Args:
            api_key: Klucz OpenAI. Jesli None, brany z env OPENAI_API_KEY.
            model: Nazwa modelu OpenAI. Domyslnie tani gpt-4o-mini.
            df: DataFrame z dialogami (z full_analysis.csv). Opcjonalny ale
                jesli jest podany, prompt bedzie bogatszy o korpus i statystyki.

        Raises:
            RuntimeError: Jesli klucz OpenAI nie zostal znaleziony.
        """
        try:
            self.client = OpenAI(api_key=api_key)
        except OpenAIError as e:
            raise RuntimeError(
                "Brak kluczy OpenAI. Ustaw OPENAI_API_KEY w .env ."
            ) from e
        self.model = model
        if df is not None:
            self.korpus = korpus_postaci(df)
        else:
            self.korpus = {}
        self.stats = {}
        for postac in self.korpus:
            self.stats[postac] = statystyki(self.korpus[postac])

    def _prompt(self, postac):
        """Buduje system prompt dla danej postaci.

        Skleja: opis postaci + style notes + przyklady ze skryptu +
        statystyki stylometryczne + regulki (nie wychodz z roli, max 1-4
        linie itp.).

        Args:
            postac: Nazwa postaci (musi byc w CHARACTER_PROFILES).

        Returns:
            str: Gotowy system prompt do podstawienia w API call.
        """
        profil = CHARACTER_PROFILES.get(postac, {})
        st = self.stats.get(postac, {})
        probka = self.korpus.get(postac, [])[:5]

        przyklady = profil.get('example_lines', []) + probka
        przyklady = przyklady[:8]
        linie_przykl = []
        for l in przyklady:
            linie_przykl.append('- "' + l + '"')
        przyklady_str = "\n".join(linie_przykl)

        opis = profil.get('description', 'Unknown character')
        notki = profil.get('style_notes', '')

        system = f"""You are a creative writer tasked with generating dialogue in the style of {postac} from Half-Life 2.

        CHARACTER PROFILE:
        {opis}

        STYLE NOTES:
        {notki}

        EXAMPLE LINES FROM THE SCRIPT:
        {przyklady_str}

        """
        if st:
            system += f"""STYLOMETRIC DATA (from corpus analysis):
                    - Average words per line: {st.get('avg_words_per_line', '?')}
                    - Question frequency: {int(st.get('question_ratio', 0) * 100)}% of lines
                    - Exclamation frequency: {int(st.get('exclamation_ratio', 0) * 100)}% of lines
                    - Ellipsis usage: {int(st.get('ellipsis_ratio', 0) * 100)}% of lines
                    - Characteristic words: {', '.join(st.get('top_words', [])[:6])}
                    """

        system += """
                RULES:
                1. Stay completely in character. Never break the fourth wall.
                2. Match the stylometric profile above — sentence length, punctuation habits, vocabulary.
                3. Gordon Freeman is always silent; react to him as the character would.
                4. Keep responses concise: 1-4 lines of dialogue maximum.
                5. Do NOT use quotation marks. Output only the spoken words.
                """

        return system

    def generuj(self, postac, sytuacja, sentyment="neutral", kontekst=""):
        """Generuje jedna wypowiedz danej postaci w danej sytuacji.

        Args:
            postac: Nazwa postaci (musi byc w CHARACTER_PROFILES).
            sytuacja: Krotki opis tego, co sie dzieje (po angielsku).
            sentyment: Sugerowany ton ("positive", "negative", "neutral").
            kontekst: Opcjonalny dodatkowy kontekst tematyczny.

        Returns:
            str: Wygenerowana linia dialogowa (bez cudzyslowow).

        Raises:
            ValueError: Jesli postac nie ma profilu.
        """
        if postac not in CHARACTER_PROFILES:
            dostepne = list(CHARACTER_PROFILES.keys())
            raise ValueError("Nieznana postac '" + postac + "'. Dostepne: " + str(dostepne))

        system = self._prompt(postac)

        user = "Situation: " + sytuacja
        if sentyment != "neutral":
            user += "\nTone: " + sentyment
        if kontekst:
            user += "\nTopic context: " + kontekst
        user += "\n\nGenerate one line of dialogue as " + postac + ":"

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        wynik = response.choices[0].message.content or ""
        return wynik.strip()

    def generuj_scene(self, postacie, sytuacja, n=4):
        """Generuje krotka scene dialogowa - n wymian na zmiane.

        Postacie mowia po kolei (round-robin). Kazda kolejna postac widzi
        poprzednie wypowiedzi w kontekscie, wiec scena ma "ciaglosc".

        Args:
            postacie: Lista nazw postaci do scene.
            sytuacja: Krotki opis sytuacji.
            n: Liczba wymian dialogowych.

        Returns:
            list[dict]: Lista {"character": str, "line": str}.
        """
        scena = []
        kontekst = "Scene: " + sytuacja + "\n\n"

        for i in range(n):
            # postacie sie zmieniaja na zmiane
            postac = postacie[i % len(postacie)]
            prompt = kontekst + "Now " + postac + " speaks:"
            linia = self.generuj(postac, prompt)
            scena.append({"character": postac, "line": linia})
            kontekst += postac + ": " + linia + "\n"

        return scena

    def generuj_wiele(self, zapytania):
        """Generuje wiele linii naraz (np. do batch ewaluacji).

        Bledy nie przerywaja petli - zostaja zapisane w polu "status"
        konkretnego wyniku.

        Args:
            zapytania: Lista slownikow z kluczami:
                "character" (str), "situation" (str),
                "sentiment_hint" (str, opcj.), "topic_context" (str, opcj.).

        Returns:
            list[dict]: Lista wynikow - kazdy ma generated_line oraz
                status ("ok" lub "error: ...").
        """
        wyniki = []
        for z in zapytania:
            try:
                linia = self.generuj(
                    postac=z["character"],
                    sytuacja=z["situation"],
                    sentyment=z.get("sentiment_hint", "neutral"),
                    kontekst=z.get("topic_context", ""),
                )
                wynik = dict(z)
                wynik["generated_line"] = linia
                wynik["status"] = "ok"
                wyniki.append(wynik)
            except Exception as e:
                wynik = dict(z)
                wynik["generated_line"] = ""
                wynik["status"] = "error: " + str(e)
                wyniki.append(wynik)

        return wyniki

    def porownaj_styl(self, postac, wygenerowane):
        """Porownuje stylometrie oryginalnych i wygenerowanych linii.

        Sluzy jako prosta forma ewaluacji generatora - im mniejsze diff
        w avg_words/question_ratio/itp. i im wieksze keyword_overlap,
        tym lepiej generator naśladuje styl postaci.

        Args:
            postac: Nazwa postaci (musi miec dane w self.stats).
            wygenerowane: Lista linii wygenerowanych przez generuj()/itd.

        Returns:
            dict: Pary _original / _generated / _diff dla kazdej metryki
                plus keyword_overlap. Lub {"error": ...} gdy brak danych.
        """
        orig = self.stats.get(postac, {})
        nowe = statystyki(wygenerowane)

        if not orig or not nowe:
            return {"error": "Brak danych do porownania"}

        out = {}
        for k in ["avg_words_per_line", "question_ratio", "exclamation_ratio", "ellipsis_ratio"]:
            o = orig.get(k, 0)
            n = nowe.get(k, 0)
            out[k + "_original"] = o
            out[k + "_generated"] = n
            out[k + "_diff"] = round(abs(o - n), 3)

        slowa_o = set(orig.get("top_words", []))
        slowa_n = set(nowe.get("top_words", []))
        if len(slowa_o) > 0:
            pokrycie = len(slowa_o & slowa_n) / len(slowa_o)
        else:
            pokrycie = 0
        out["keyword_overlap"] = round(pokrycie, 3)

        return out


if __name__ == "__main__":
    import os

    df = None
    if Path("data/full_analysis.csv").exists():
        df = pd.read_csv("data/full_analysis.csv")

    gen = Generator(
        api_key=os.environ.get("OPENAI_API_KEY"),
        df=df,
    )

    demo = [
        ("G-Man", "Gordon Freeman has just completed his mission", "neutral"),
        ("Alyx", "Gordon just saved her from a combine soldier", "positive"),
        ("Dr. Breen", "Citizens are rebelling in the streets", "negative"),
        ("Barney", "He spots a headcrab in the ventilation shaft", "negative"),
        ("Father Gregori", "A new survivor arrives in Ravenholm at night", "neutral"),
        ("Vortigaunt", "The resistance has won a major battle", "positive"),
    ]

    for postac, sytuacja, sent in demo:
        print("\n[" + postac + "]")
        print("Sytuacja:", sytuacja)
        linia = gen.generuj(postac, sytuacja, sent)
        print("Wypowiedz:", linia)

    scena = gen.generuj_scene(
        postacie=["Alyx", "Barney", "Dr. Kleiner"],
        sytuacja="They have just escaped from the Citadel and are regrouping",
        n=6,
    )
    for w in scena:
        print(w['character'] + ":", w['line'])
