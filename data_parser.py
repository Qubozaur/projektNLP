import re
import json
from pathlib import Path
import pdfplumber

# nazwy rozdzialow w grze
ROZDZIALY = {
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

# w skrypcie postacie sa pisane roznie, tu sprowadzam do jednej nazwy
ALIASY = {
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

GLOWNE_POSTACIE = {"G-Man", "Barney", "Dr. Kleiner", "Alyx", "Eli", "Dr. Breen", "Dr. Mossman", "Father Gregori", "Colonel Cubbage", "Vortigaunt"}


class Dialog:
    """Jedna wypowiedz postaci wyciagnieta ze skryptu.

    Attributes:
        chapter_id: Identyfikator rozdzialu, np. "2a".
        chapter_name: Pelna nazwa rozdzialu z gry.
        character: Znormalizowana nazwa postaci (po przejsciu przez ALIASY).
        is_main_character: True jesli postac nalezy do GLOWNE_POSTACIE.
        text: Tekst wypowiedzi (juz oczyszczony).
        line_index: Globalny numer linii w skrypcie (kolejnosc wystapienia).
    """

    def __init__(self, chapter_id, chapter_name, character, is_main_character, text, line_index):
        self.chapter_id = chapter_id
        self.chapter_name = chapter_name
        self.character = character
        self.is_main_character = is_main_character
        self.text = text
        self.line_index = line_index


class Stage:
    """Stage direction (didaskalia), czyli opis sceny w nawiasach okraglych."""

    def __init__(self, chapter_id, text):
        self.chapter_id = chapter_id
        self.text = text


class Parser:
    """Parser PDF -> lista obiektow Dialog / Stage.

    Pipeline:
        1. wczytaj_pdf()         - PDF -> jeden duzy string
        2. podziel_na_rozdzialy()- string -> dict {chapter_id: tekst}
        3. _parsuj_rozdzial()    - kazdy rozdzial -> linie dialogowe
        4. zapisz()              - wynik do data/dialogues.json

    Wygodnie wywolac od razu metode parsuj(), ktora robi 1-3.
    """
    RE_DIALOG = re.compile(r'^([A-Za-z0-9 ./\-]+?)\s*[-–]\s*(.+)$')
    RE_STAGE = re.compile(r'^\((.+)\)$', re.DOTALL)
    RE_ROZDZIAL = re.compile(r'^\s*(2[a-n])\.\s+(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
    RE_STRONA = re.compile(r'^--\s*\d+\s+of\s+\d+\s*--$', re.IGNORECASE)
    RE_TYTUL = re.compile(r'^half-life\s*2\s*script$', re.IGNORECASE)
    RE_SPIS = re.compile(r'^(table of contents|[12]\.\s+\w+|2[a-n]\.\s+.+)$', re.IGNORECASE)

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        self.tekst = ""
        self.dialogi = []
        self.stage_dir = []

    def wczytaj_pdf(self):
        """Wczytuje caly PDF do jednego stringa (self.tekst).

        Returns:
            str: Polaczony tekst ze wszystkich stron PDFa.
        """
        with pdfplumber.open(self.pdf_path) as pdf:
            strony = []
            for strona in pdf.pages:
                t = strona.extract_text() or ""
                strony.append(t)
        self.tekst = "\n".join(strony)
        return self.tekst

    def _smieci(self, linia):
        l = linia.strip()
        if not l:
            return True
        if self.RE_STRONA.match(l):
            return True
        if self.RE_TYTUL.match(l):
            return True
        if self.RE_SPIS.match(l):
            return True
        return False

    def _oczysc(self, tekst):
        t = re.sub(r'\s+', ' ', tekst).strip()
        t = t.replace(" .", ".")
        t = t.replace(" ,", ",")
        t = t.replace(" !", "!")
        t = t.replace(" ?", "?")
        return t

    def podziel_na_rozdzialy(self):
        """Dzieli surowy tekst PDF na rozdzialy 2a-2n po naglowkach.

        Pomija spis tresci na poczatku (Table of contents).

        Returns:
            dict[str, str]: Mapa {chapter_id: tekst rozdzialu}.
        """
        rozdzialy = {}
        separatory = []
        start = 0

        if "Table Of contents" in self.tekst:
            koniec_spisu = 0
            for m in self.RE_ROZDZIAL.finditer(self.tekst):
                if m.start() < 5000:
                    koniec_spisu = m.end()
                else:
                    break
            if koniec_spisu:
                start = koniec_spisu

        for m in self.RE_ROZDZIAL.finditer(self.tekst):
            if m.start() < start:
                continue
            id_r = m.group(1).lower()
            nazwa = m.group(2).strip()
            separatory.append((m.start(), m.end(), id_r, nazwa))

        if separatory:
            przed = self.tekst[start:separatory[0][0]].strip()
            if przed:
                pierwszy_id = separatory[0][2]
                if pierwszy_id != "2a":
                    rozdzialy["2a"] = przed
                else:
                    rozdzialy["intro"] = przed

        for i in range(len(separatory)):
            poczatek = separatory[i][1]
            id_r = separatory[i][2]
            if i + 1 < len(separatory):
                koniec = separatory[i + 1][0]
            else:
                koniec = len(self.tekst)
            rozdzialy[id_r] = self.tekst[poczatek:koniec]

        if not separatory:
            if start:
                rozdzialy["2a"] = self.tekst[start:]
            else:
                rozdzialy["2a"] = self.tekst

        return rozdzialy

    def _parsuj_rozdzial(self, chapter_id, tekst_rozdzialu, licznik):
        """Wyciaga z tekstu rozdzialu linie Dialog i Stage.

        Linia dialogowa ma format "Postac - tekst". Wieloliniowe wypowiedzi
        sa scalane do jednej. Stage directions to wszystko w nawiasach ().
        Wyniki dopisuje do self.dialogi / self.stage_dir.

        Args:
            chapter_id: ID rozdzialu, np. "2a".
            tekst_rozdzialu: Surowy tekst tego rozdzialu.
            licznik: Lista jednoelementowa z aktualnym numerem linii
                (uzywana zamiast int, zeby przekazywac przez referencje).
        """
        nazwa = ROZDZIALY.get(chapter_id, chapter_id)
        linie = tekst_rozdzialu.split('\n')
        i = 0
        while i < len(linie):
            linia = linie[i].strip()
            if not linia:
                i += 1
                continue
            if self._smieci(linia):
                i += 1
                continue

            if linia.startswith('('):
                cala = linia
                while not cala.endswith(')') and i + 1 < len(linie):
                    i += 1
                    cala += ' ' + linie[i].strip()
                srodek = self._oczysc(cala.strip('()'))
                if srodek and not self._smieci(srodek):
                    self.stage_dir.append(Stage(chapter_id, srodek))
                i += 1
                continue

            m = self.RE_DIALOG.match(linia)
            if m:
                raw_postac = m.group(1).strip().lower()
                tekst = m.group(2).strip()
                while (i + 1 < len(linie)
                       and linie[i + 1].strip()
                       and not self.RE_DIALOG.match(linie[i + 1].strip())
                       and not linie[i + 1].strip().startswith('(')
                       and not linie[i + 1].strip().startswith('=')
                       and not self._smieci(linie[i + 1].strip())):
                    i += 1
                    tekst += ' ' + linie[i].strip()
                tekst = self._oczysc(tekst)
                if not tekst or self._smieci(tekst):
                    i += 1
                    continue
                postac = self._postac(raw_postac)
                czy_glowna = postac in GLOWNE_POSTACIE
                d = Dialog(
                    chapter_id=chapter_id,
                    chapter_name=nazwa,
                    character=postac,
                    is_main_character=czy_glowna,
                    text=tekst,
                    line_index=licznik[0],
                )
                self.dialogi.append(d)
                licznik[0] += 1
            i += 1

    def _postac(self, raw):
        r = raw.lower().strip()
        if r in ALIASY:
            return ALIASY[r]
        for alias in ALIASY:
            if r.startswith(alias):
                return ALIASY[alias]
        return raw.title().strip()

    def parsuj(self):
        """Pelny pipeline: wczytaj PDF, podziel na rozdzialy, sparsuj wszystko.

        Returns:
            tuple[list[Dialog], list[Stage]]: Lista dialogow i stage directions.
        """
        self.wczytaj_pdf()
        rozdzialy = self.podziel_na_rozdzialy()
        licznik = [0]
        for id_r in rozdzialy:
            tekst_r = rozdzialy[id_r]
            self._parsuj_rozdzial(id_r, tekst_r, licznik)
        print("Wyciagnieto", len(self.dialogi), "linii dialogowych")
        print("Wyciagnieto", len(self.stage_dir), "stage directions")
        return self.dialogi, self.stage_dir

    def zapisz(self, out_path="data/dialogues.json"):
        """Zapisuje sparsowane dane do pliku JSON.

        Args:
            out_path: Sciezka docelowa. Domyslnie data/dialogues.json.
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        lista_dial = []
        for d in self.dialogi:
            lista_dial.append(d.__dict__)
        lista_stage = []
        for s in self.stage_dir:
            lista_stage.append(s.__dict__)
        dane = {
            "dialogues": lista_dial,
            "stage_directions": lista_stage,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dane, f, ensure_ascii=False, indent=2)
        print("Zapisano do", out_path)


if __name__ == "__main__":
    p = Parser("data/half_life_2_script.pdf")
    dialogi, stage = p.parsuj()
    p.zapisz("data/dialogues.json")
