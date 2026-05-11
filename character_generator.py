import json
import re
from openai import OpenAI
from openai import OpenAIError
from pathlib import Path
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

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


def build_character_corpus(df: pd.DataFrame) -> dict[str, list[str]]:
    corpus = defaultdict(list)
    for _, row in df.iterrows():
        if row["character"] in CHARACTER_PROFILES:
            corpus[row["character"]].append(row["text"])
    return dict(corpus)


def get_style_stats(lines: list[str]) -> dict:
    if not lines:
        return {}

    avg_words = sum(len(l.split()) for l in lines) / len(lines)
    question_ratio = sum(1 for l in lines if "?" in l) / len(lines)
    exclaim_ratio = sum(1 for l in lines if "!" in l) / len(lines)
    ellipsis_ratio = sum(1 for l in lines if "..." in l) / len(lines)

    stopwords = {"the", "a", "an", "i", "you", "we", "he", "she", "it",
                 "to", "of", "and", "in", "is", "have", "that", "this",
                 "are", "was", "be", "not", "with", "for", "my", "your"}
    word_freq: dict[str, int] = defaultdict(int)
    for line in lines:
        for word in re.findall(r"\b\w+\b", line.lower()):
            if word not in stopwords and len(word) > 3:
                word_freq[word] += 1

    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:10]

    return {
        "avg_words_per_line": round(avg_words, 1),
        "question_ratio": round(question_ratio, 2),
        "exclamation_ratio": round(exclaim_ratio, 2),
        "ellipsis_ratio": round(ellipsis_ratio, 2),
        "top_words": [w for w, _ in top_words],
        "total_lines": len(lines),
    }


class CharacterGenerator:

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        df: pd.DataFrame | None = None,
    ):
        try:
            self.client = OpenAI(api_key=api_key)
        except OpenAIError as e:
            raise RuntimeError(
                "Missing OpenAI credentials. Set OPENAI_API_KEY in .env or pass api_key explicitly."
            ) from e
        self.model = model
        self.corpus = build_character_corpus(df) if df is not None else {}
        self.style_stats = {
            char: get_style_stats(lines)
            for char, lines in self.corpus.items()
        }

    def _build_system_prompt(self, character: str) -> str:
        profile = CHARACTER_PROFILES.get(character, {})
        stats = self.style_stats.get(character, {})
        corpus_sample = self.corpus.get(character, [])[:5]

        system = f"""You are a creative writer tasked with generating dialogue in the style of {character} from Half-Life 2.

CHARACTER PROFILE:
{profile.get('description', 'Unknown character')}

STYLE NOTES:
{profile.get('style_notes', '')}

EXAMPLE LINES FROM THE SCRIPT:
{chr(10).join(f'- "{line}"' for line in (profile.get('example_lines', []) + corpus_sample)[:8])}

"""
        if stats:
            system += f"""STYLOMETRIC DATA (from corpus analysis):
- Average words per line: {stats.get('avg_words_per_line', '?')}
- Question frequency: {int(stats.get('question_ratio', 0) * 100)}% of lines
- Exclamation frequency: {int(stats.get('exclamation_ratio', 0) * 100)}% of lines
- Ellipsis usage: {int(stats.get('ellipsis_ratio', 0) * 100)}% of lines
- Characteristic words: {', '.join(stats.get('top_words', [])[:6])}
"""

        system += """
RULES:
1. Stay completely in character. Never break the fourth wall.
2. Match the stylometric profile above — sentence length, punctuation habits, vocabulary.
3. Gordon Freeman is always silent; react to him as the character would.
4. Keep responses concise: 1-4 lines of dialogue maximum.
5. Do NOT use quotation marks. Output only the spoken words."""

        return system

    def generate(
        self,
        character: str,
        situation: str,
        sentiment_hint: str = "neutral",
        topic_context: str = "",
    ) -> str:
        if character not in CHARACTER_PROFILES:
            available = list(CHARACTER_PROFILES.keys())
            raise ValueError(f"Nieznana postać '{character}'. Dostępne: {available}")

        system = self._build_system_prompt(character)

        user_prompt = f"Situation: {situation}"
        if sentiment_hint != "neutral":
            user_prompt += f"\nTone: {sentiment_hint}"
        if topic_context:
            user_prompt += f"\nTopic context: {topic_context}"
        user_prompt += f"\n\nGenerate one line of dialogue as {character}:"

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )

        return (response.choices[0].message.content or "").strip()

    def generate_scene(
        self,
        characters: list[str],
        situation: str,
        n_exchanges: int = 4,
    ) -> list[dict]:
        scene: list[dict] = []
        context = f"Scene: {situation}\n\n"

        for i in range(n_exchanges):
            character = characters[i % len(characters)]
            prompt = context + f"Now {character} speaks:"

            line = self.generate(character, prompt)
            scene.append({"character": character, "line": line})
            context += f"{character}: {line}\n"

        return scene

    def batch_generate(
        self,
        requests: list[dict],
    ) -> list[dict]:
        results = []
        for req in requests:
            try:
                line = self.generate(
                    character=req["character"],
                    situation=req["situation"],
                    sentiment_hint=req.get("sentiment_hint", "neutral"),
                    topic_context=req.get("topic_context", ""),
                )
                results.append({**req, "generated_line": line, "status": "ok"})
            except Exception as e:
                results.append({**req, "generated_line": "", "status": f"error: {e}"})

        return results

    def stylometric_eval(
        self,
        character: str,
        generated_lines: list[str],
    ) -> dict:
        original_stats = self.style_stats.get(character, {})
        generated_stats = get_style_stats(generated_lines)

        if not original_stats or not generated_stats:
            return {"error": "Brak danych do porównania"}

        comparison = {}
        for key in ["avg_words_per_line", "question_ratio", "exclamation_ratio", "ellipsis_ratio"]:
            orig = original_stats.get(key, 0)
            gen = generated_stats.get(key, 0)
            diff = abs(orig - gen)
            comparison[f"{key}_original"] = orig
            comparison[f"{key}_generated"] = gen
            comparison[f"{key}_diff"] = round(diff, 3)

        orig_words = set(original_stats.get("top_words", []))
        gen_words = set(generated_stats.get("top_words", []))
        overlap = len(orig_words & gen_words) / max(len(orig_words), 1)
        comparison["keyword_overlap"] = round(overlap, 3)

        return comparison


if __name__ == "__main__":
    import os

    df = None
    if Path("data/full_analysis.csv").exists():
        df = pd.read_csv("data/full_analysis.csv")

    gen = CharacterGenerator(
        api_key=os.environ.get("OPENAI_API_KEY"),
        df=df,
    )

    print("═" * 60)
    print("GENEROWANIE WYPOWIEDZI W STYLU POSTACI HL2")
    print("═" * 60)

    demos = [
        ("G-Man", "Gordon Freeman has just completed his mission", "neutral"),
        ("Alyx", "Gordon just saved her from a combine soldier", "positive"),
        ("Dr. Breen", "Citizens are rebelling in the streets", "negative"),
        ("Barney", "He spots a headcrab in the ventilation shaft", "negative"),
        ("Father Gregori", "A new survivor arrives in Ravenholm at night", "neutral"),
        ("Vortigaunt", "The resistance has won a major battle", "positive"),
    ]

    for character, situation, sentiment in demos:
        print(f"\n[{character}]")
        print(f"Sytuacja: {situation}")
        line = gen.generate(character, situation, sentiment)
        print(f"Wypowiedź: {line}")

    print("\n─── Scena dialogowa ───")
    scene = gen.generate_scene(
        characters=["Alyx", "Barney", "Dr. Kleiner"],
        situation="They have just escaped from the Citadel and are regrouping",
        n_exchanges=6,
    )
    for exchange in scene:
        print(f"{exchange['character']}: {exchange['line']}")
