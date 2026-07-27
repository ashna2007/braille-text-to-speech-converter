"""Cross-platform Liblouis integration for Braille back-translation."""

from functools import lru_cache
import os
from pathlib import Path
import shutil
import string
import subprocess
import sys


DISPLAY_TABLE = "unicode.dis"
GRADE1_TABLE = "en-us-g1.ctb"
GRADE2_TABLE = "en-us-g2.ctb"
SPACE_CELL = "\u2800"


class LiblouisUnavailableError(RuntimeError):
    """Raised when the Liblouis command-line tools cannot be located."""


def _executable_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("lou_translate.exe", "lou_translate")
    return ("lou_translate",)


def _executable_candidates() -> list[Path]:
    candidates: list[Path] = []

    configured_executable = os.environ.get("LIBLOUIS_TRANSLATE")
    if configured_executable:
        candidates.append(Path(configured_executable).expanduser())

    configured_home = os.environ.get("LIBLOUIS_HOME")
    if configured_home:
        home = Path(configured_home).expanduser()
        candidates.extend(home / "bin" / name for name in _executable_names())

    for name in _executable_names():
        discovered = shutil.which(name)
        if discovered:
            candidates.append(Path(discovered))

    # Common package-manager locations. PATH discovery above remains preferred.
    if sys.platform == "darwin":
        candidates.extend(
            [
                Path("/opt/homebrew/bin/lou_translate"),
                Path("/usr/local/bin/lou_translate"),
            ]
        )
    elif os.name == "nt":
        candidates.append(
            Path(__file__).resolve().parent
            / "liblouis-bin"
            / "bin"
            / "lou_translate.exe"
        )
    else:
        candidates.extend(
            [
                Path("/usr/bin/lou_translate"),
                Path("/usr/local/bin/lou_translate"),
            ]
        )

    return candidates


@lru_cache(maxsize=1)
def find_lou_translate() -> Path:
    """Return the installed ``lou_translate`` executable."""

    for candidate in _executable_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    if sys.platform == "darwin":
        setup_hint = "Install it with `brew install liblouis`."
    elif os.name == "nt":
        setup_hint = (
            "Install Liblouis and set LIBLOUIS_HOME, or set "
            "LIBLOUIS_TRANSLATE to lou_translate.exe."
        )
    else:
        setup_hint = (
            "Install Liblouis with your system package manager, or set "
            "LIBLOUIS_TRANSLATE."
        )

    raise LiblouisUnavailableError(
        "Liblouis is required for contracted Braille translation but "
        f"`lou_translate` was not found. {setup_hint}"
    )


def _table_environment() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("LOUIS_TABLEPATH"):
        return env

    configured_home = env.get("LIBLOUIS_HOME")
    if configured_home:
        table_dir = (
            Path(configured_home).expanduser()
            / "share"
            / "liblouis"
            / "tables"
        )
        if table_dir.is_dir():
            env["LOUIS_TABLEPATH"] = str(table_dir)

    return env


def _run_liblouis(args: list[str], text: str) -> str:
    executable = find_lou_translate()
    try:
        result = subprocess.run(
            [str(executable), *args],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_table_environment(),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "Liblouis returned a non-zero exit code."
        raise RuntimeError(f"Liblouis translation failed: {detail}") from exc

    return result.stdout.rstrip("\r\n")


def text_to_braille(text: str, table: str = GRADE2_TABLE) -> str:
    """Translate ordinary text to Unicode Braille cells."""

    return _run_liblouis(["-d", DISPLAY_TABLE, table], text)


def braille_to_text(braille: str, table: str = GRADE2_TABLE) -> str:
    """Back-translate Unicode Braille cells to ordinary text."""

    return _run_liblouis(["-b", "-d", DISPLAY_TABLE, table], braille)


@lru_cache(maxsize=1)
def _letter_to_cell() -> dict[str, str]:
    cells = text_to_braille(string.ascii_lowercase, GRADE1_TABLE)
    if len(cells) != len(string.ascii_uppercase):
        raise RuntimeError(
            "Liblouis returned an unexpected Grade-1 alphabet mapping."
        )
    return dict(zip(string.ascii_uppercase, cells))


def letters_to_braille_cells(letters: str) -> str:
    """Convert A-Z classifier labels into their corresponding Braille cells."""

    mapping = _letter_to_cell()
    cells: list[str] = []

    for character in letters.upper():
        if character in mapping:
            cells.append(mapping[character])
        elif character == "\n":
            cells.append("\n")
        elif character.isspace():
            cells.append(SPACE_CELL)
        else:
            raise ValueError(
                f"Unsupported classifier output {character!r}; expected A-Z or whitespace."
            )

    return "".join(cells)


def translate_braille(recognized_text: str) -> str:
    """Interpret A-Z Braille-cell labels as contracted English Braille."""

    recognized_text = recognized_text.strip()
    if not recognized_text:
        return ""

    braille_cells = letters_to_braille_cells(recognized_text)
    return braille_to_text(braille_cells, GRADE2_TABLE).strip()
