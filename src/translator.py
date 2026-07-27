from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shutil
import string
import subprocess


DISPLAY_TABLE = "unicode.dis"
GRADE_TABLES = {
    1: "en-us-g1.ctb",
    2: "en-us-g2.ctb",
}
SPACE_CELL = "\u2800"


@dataclass(frozen=True)
class TranslationResult:
    raw_labels: str
    braille_cells: str
    text: str
    grade: int
    available: bool
    error: str | None = None


def _liblouis_home() -> Path | None:
    value = os.environ.get("LIBLOUIS_HOME")
    return Path(value).expanduser() if value else None


@lru_cache(maxsize=1)
def _lou_translate_path() -> Path:
    configured_path = os.environ.get("LOU_TRANSLATE")
    candidates: list[Path] = []

    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    liblouis_home = _liblouis_home()
    if liblouis_home is not None:
        candidates.extend(
            [
                liblouis_home / "bin" / "lou_translate",
                liblouis_home / "bin" / "lou_translate.exe",
            ]
        )

    discovered_path = shutil.which("lou_translate")
    if discovered_path:
        candidates.append(Path(discovered_path))

    candidates.extend(
        [
            Path("/opt/homebrew/bin/lou_translate"),
            Path("/usr/local/bin/lou_translate"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Liblouis was not found. Install it or set LOU_TRANSLATE to the "
        "lou_translate executable."
    )


def _run_liblouis(args: list[str], text: str) -> str:
    environment = os.environ.copy()
    liblouis_home = _liblouis_home()
    if liblouis_home is not None:
        table_directory = liblouis_home / "share" / "liblouis" / "tables"
        if table_directory.is_dir():
            environment["LOUIS_TABLEPATH"] = str(table_directory)

    result = subprocess.run(
        [str(_lou_translate_path()), *args],
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=True,
        timeout=10,
    )
    return result.stdout


def text_to_braille(text: str, grade: int = 1) -> str:
    table = GRADE_TABLES[grade]
    return _run_liblouis(["-d", DISPLAY_TABLE, table], text)


def braille_to_text(braille: str, grade: int = 1) -> str:
    table = GRADE_TABLES[grade]
    return _run_liblouis(["-b", "-d", DISPLAY_TABLE, table], braille)


@lru_cache(maxsize=1)
def _letter_to_cell() -> dict[str, str]:
    cells = text_to_braille(string.ascii_lowercase, grade=1).rstrip("\r\n")
    if len(cells) < len(string.ascii_uppercase):
        raise RuntimeError("Liblouis returned an incomplete A-Z Braille mapping.")
    return dict(zip(string.ascii_uppercase, cells))


def letters_to_braille_cells(letters: str) -> str:
    mapping = _letter_to_cell()
    cells: list[str] = []

    for character in letters.upper():
        if character in mapping:
            cells.append(mapping[character])
        elif character in "\r\n":
            cells.append(character)
        elif character.isspace():
            cells.append(SPACE_CELL)
        else:
            raise ValueError(f"Unsupported classifier output: {character!r}")

    return "".join(cells)


def translate_braille(recognized_text: str, grade: int = 1) -> TranslationResult:
    raw_labels = recognized_text.strip()
    if not raw_labels:
        return TranslationResult(
            raw_labels="",
            braille_cells="",
            text="",
            grade=grade,
            available=True,
        )

    try:
        braille_cells = letters_to_braille_cells(raw_labels)
        translated_text = braille_to_text(braille_cells, grade=grade).strip()
        return TranslationResult(
            raw_labels=raw_labels,
            braille_cells=braille_cells,
            text=translated_text,
            grade=grade,
            available=True,
        )
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        return TranslationResult(
            raw_labels=raw_labels,
            braille_cells="",
            text=raw_labels,
            grade=grade,
            available=False,
            error=str(exc),
        )
