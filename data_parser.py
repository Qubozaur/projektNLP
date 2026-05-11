import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import pdfplumber 

CHAPTERS = {
    "2a": "Point Insertion",
    "2b": "A Red Letter Day",
    "2c": "Route Kanal",
    "2d": "Water Hazard",
    "2e": "Black Mesa East",
    "2f": "We Don't Go To Ravenholm",
    "2g": "Highway 17",
    "2h": "Sandtraps",
    "2i": "Nova Prospekt",
    "2j": "Entanglement",
    "2k": "Anti-citizen One",
    "2l": "Follow Freeman!",
    "2m": "Our Benefactors",
    "2n": "Dark Energy",
}

CHARACTER_ALIASES = {
    "gman": "G-Man",
    "g-man": "G-Man",
    "barney": "Barney",
    "dr. kliener": "Dr. Kleiner",
    "dr. kleiner": "Dr. Kleiner",
    "dr. kiener": "Dr. Kleiner",
    "kleiner": "Dr. Kleiner",
    "kliener": "Dr. Kleiner",
    "kiener": "Dr. Kleiner",
    "alyx": "Alyx",
    "aylx": "Alyx", 
    "eli": "Eli",
    "dr. breen": "Dr. Breen",
    "breen": "Dr. Breen",
    "monitor/breen": "Dr. Breen",
    "judith": "Dr. Mossman",
    "dr. mossman": "Dr. Mossman",
    "father gregori": "Father Gregori",
    "colonel cubbage": "Colonel Cubbage",
    "vorigaunt": "Vortigaunt",
    "voriagunt": "Vortigaunt",
    "citizen": "Citizen",
    "civil patrol": "Civil Patrol",
    "resistance": "Resistance",
    "woman on radio": "Radio",
    "man on radio": "Radio",
    "alyx (radio)": "Alyx",
    "warning voice": "Announcer",
}

MAIN_CHARACTERS = {"G-Man", "Barney", "Dr. Kleiner", "Alyx", "Eli", "Dr. Breen", "Dr. Mossman", "Father Gregori", "Colonel Cubbage", "Vortigaunt"}

@dataclass
class DialogueLine:
    chapter_id: str
    chapter_name: str
    character: str
    is_main_character: bool
    text: str
    line_index: int       


@dataclass
class StageDirection:
    chapter_id: str
    text: str


class HL2ScriptParser:
    DIALOGUE_RE = re.compile(r'^([A-Za-z0-9 ./\-]+?)\s*[-–]\s*(.+)$')
    STAGE_RE = re.compile(r'^\((.+)\)$', re.DOTALL)
    CHAPTER_SEP_RE = re.compile(r'^\s*(2[a-n])\.\s+(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
    PAGE_MARKER_RE = re.compile(r'^--\s*\d+\s+of\s+\d+\s*--$', re.IGNORECASE)
    SCRIPT_TITLE_RE = re.compile(r'^half-life\s*2\s*script$', re.IGNORECASE)
    TOC_LINE_RE = re.compile(r'^(table of contents|[12]\.\s+\w+|2[a-n]\.\s+.+)$', re.IGNORECASE)

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.raw_text = ""
        self.dialogues: list[DialogueLine] = []
        self.stage_directions: list[StageDirection] = []

    def extract_text(self) -> str:
        with pdfplumber.open(self.pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        self.raw_text = "\n".join(pages)
        return self.raw_text

    def _is_noise_line(self, line: str) -> bool:
        candidate = line.strip()
        if not candidate:
            return True
        if self.PAGE_MARKER_RE.match(candidate):
            return True
        if self.SCRIPT_TITLE_RE.match(candidate):
            return True
        return bool(self.TOC_LINE_RE.match(candidate))

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        cleaned = cleaned.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
        return cleaned

    def split_chapters(self) -> dict[str, str]:
        chapters: dict[str, str] = {}
        separators = []
        script_start = 0
        if "Table Of contents" in self.raw_text:
            toc_end = 0
            for m in self.CHAPTER_SEP_RE.finditer(self.raw_text):
                if m.start() < 5000:
                    toc_end = m.end()
                else:
                    break
            if toc_end:
                script_start = toc_end
        for m in self.CHAPTER_SEP_RE.finditer(self.raw_text):
            if m.start() < script_start:
                continue
            sep_id = m.group(1).lower()       
            sep_name = m.group(2).strip()
            separators.append((m.start(), m.end(), sep_id, sep_name))
        if separators:
            preface = self.raw_text[script_start:separators[0][0]].strip()
            if preface:
                first_id = separators[0][2]
                missing_intro_id = "2a" if first_id != "2a" else "intro"
                chapters[missing_intro_id] = preface
        for i, (start, end, ch_id, _) in enumerate(separators):
            next_start = separators[i + 1][0] if i + 1 < len(separators) else len(self.raw_text)
            chapters[ch_id] = self.raw_text[end:next_start]
        if not separators:
            chapters["2a"] = self.raw_text[script_start:] if script_start else self.raw_text
        return chapters

    def _parse_chapter(self, chapter_id: str, chapter_text: str, line_counter: list):
        chapter_name = CHAPTERS.get(chapter_id, chapter_id)
        lines = chapter_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if self._is_noise_line(line):
                i += 1
                continue
            if line.startswith('('):
                full = line
                while not full.endswith(')') and i + 1 < len(lines):
                    i += 1
                    full += ' ' + lines[i].strip()
                inner = self._clean_text(full.strip('()'))
                if inner and not self._is_noise_line(inner):
                    self.stage_directions.append(StageDirection(chapter_id, inner))
                i += 1
                continue
            m = self.DIALOGUE_RE.match(line)
            if m:
                raw_char = m.group(1).strip().lower()
                text = m.group(2).strip()
                while (i + 1 < len(lines)
                       and lines[i + 1].strip()
                       and not self.DIALOGUE_RE.match(lines[i + 1].strip())
                       and not lines[i + 1].strip().startswith('(')
                       and not lines[i + 1].strip().startswith('=')
                       and not self._is_noise_line(lines[i + 1].strip())):
                    i += 1
                    text += ' ' + lines[i].strip()
                text = self._clean_text(text)
                if not text or self._is_noise_line(text):
                    i += 1
                    continue
                character = self._resolve_character(raw_char)
                is_main = character in MAIN_CHARACTERS
                self.dialogues.append(DialogueLine(
                    chapter_id=chapter_id,
                    chapter_name=chapter_name,
                    character=character,
                    is_main_character=is_main,
                    text=text,
                    line_index=line_counter[0],
                ))
                line_counter[0] += 1
            i += 1

    def _resolve_character(self, raw: str) -> str:
        raw_lower = raw.lower().strip()
        if raw_lower in CHARACTER_ALIASES:
            return CHARACTER_ALIASES[raw_lower]
        for alias, canonical in CHARACTER_ALIASES.items():
            if raw_lower.startswith(alias):
                return canonical
        return raw.title().strip()

    def parse(self) -> tuple[list[DialogueLine], list[StageDirection]]:
        self.extract_text()
        chapters = self.split_chapters()
        counter = [0]
        for ch_id, ch_text in chapters.items():
            self._parse_chapter(ch_id, ch_text, counter)
        print(f"[Parser] Wyodrębniono {len(self.dialogues)} linii dialogowych")
        print(f"[Parser] Wyodrębniono {len(self.stage_directions)} stage directions")
        return self.dialogues, self.stage_directions

    def save_json(self, out_path: str = "data/dialogues.json"):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dialogues": [asdict(d) for d in self.dialogues],
            "stage_directions": [asdict(s) for s in self.stage_directions],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Parser] Zapisano do {out_path}")

if __name__ == "__main__":
    parser = HL2ScriptParser("data/half_life_2_script.pdf")
    dialogues, stage_dirs = parser.parse()
    parser.save_json("data/dialogues.json")