from __future__ import annotations

import csv
import importlib
import io
import json
import mimetypes
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import yaml


# Optional deps are imported lazily so the module works even if not installed.
def _try_import(mod):
    try:
        return __import__(mod)
    except Exception:
        return None


_pdfminer = None
_docx = None
_pptx = None
_openpyxl = None
_pytesseract = None
_PIL_Image = None

TEXT_EXTS = {
    ".txt",
    ".md",
    ".rst",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".py",
    ".ipynb",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".kt",
    ".kts",
    ".cs",
    ".vb",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".hh",
    ".go",
    ".rs",
    ".swift",
    ".m",
    ".mm",
    ".php",
    ".rb",
    ".pl",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".scala",
    ".clj",
    ".edn",
    ".lua",
}

CSV_EXTS = {".csv"}
TSV_EXTS = {".tsv"}
XLSX_EXTS = {".xlsx"}
DOCX_EXTS = {".docx"}
PPTX_EXTS = {".pptx"}
PDF_EXTS = {".pdf"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
JSON_EXTS = {".json"}
YAML_EXTS = {".yaml", ".yml"}
XML_EXTS = {".xml"}

MAX_TEXT_BYTES = 10 * 1024 * 1024  # 10MB safety limit


def extract_text_from_path(path: Path, ocr: bool = False) -> str:
    """
    Extract text from a single file path. Returns best-effort text.
    Never raises on unsupported types; returns an explanatory message instead.
    """
    ext = path.suffix.lower()

    if ext in TEXT_EXTS:
        return _read_text_file(path)
    if ext in CSV_EXTS:
        return _read_csv(path, dialect="excel")
    if ext in TSV_EXTS:
        return _read_csv(path, dialect="excel-tab")
    if ext in XLSX_EXTS:
        return _read_xlsx(path)
    if ext in DOCX_EXTS:
        return _read_docx(path)
    if ext in PPTX_EXTS:
        return _read_pptx(path)
    if ext in PDF_EXTS:
        return _read_pdf(path)
    if ext in IMG_EXTS:
        return _read_image(path, ocr=ocr)
    if ext in JSON_EXTS:
        return _read_json(path)
    if ext in YAML_EXTS:
        return _read_yaml(path)
    if ext in XML_EXTS:
        return _read_xml(path)

    try:
        if path.stat().st_size <= MAX_TEXT_BYTES:
            return _read_text_file(path)
    except Exception:
        pass

    mime, _ = mimetypes.guess_type(path.name)
    if mime and mime.startswith("text/"):
        return _read_text_file(path)

    return f"[unsupported] {path.name} (.{ext.lstrip('.')}) — no extractor available.\n"


def _read_text_file(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[error] reading text file {path.name}: {e}\n"


def _read_csv(path: Path, dialect: str) -> str:
    out = io.StringIO()
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, dialect=dialect)
            for row in reader:
                out.write("\t".join(row) + "\n")
    except Exception as e:
        return f"[error] reading CSV/TSV {path.name}: {e}\n"
    return out.getvalue()


def _read_xlsx(path: Path) -> str:
    global _openpyxl
    if _openpyxl is None:
        _openpyxl = _try_import("openpyxl")
    if not _openpyxl:
        return f"[missing] openpyxl not installed for {path.name}\n"
    try:
        wb = _openpyxl.load_workbook(path, data_only=True, read_only=True)
        out = io.StringIO()
        for ws in wb.worksheets:
            out.write(f"# Sheet: {ws.title}\n")
            for row in ws.iter_rows(values_only=True):
                cells = ["" if v is None else str(v) for v in row]
                out.write("\t".join(cells) + "\n")
        return out.getvalue()
    except Exception as e:
        return f"[error] reading XLSX {path.name}: {e}\n"


def _read_docx(path: Path) -> str:
    global _docx
    if _docx is None:
        _docx = _try_import("docx")
    if not _docx:
        return f"[missing] python-docx not installed for {path.name}\n"
    try:
        doc = _docx.Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text]
        for tbl in doc.tables:
            for row in tbl.rows:
                cells = [c.text for c in row.cells]
                paras.append("\t".join(cells))
        return "\n".join(paras) + "\n"
    except Exception as e:
        return f"[error] reading DOCX {path.name}: {e}\n"


def _read_pptx(path: Path) -> str:
    global _pptx
    if _pptx is None:
        _pptx = _try_import("pptx")
    if not _pptx:
        return f"[missing] python-pptx not installed for {path.name}\n"
    try:
        prs = _pptx.Presentation(str(path))
        out_lines = []
        for idx, slide in enumerate(prs.slides, start=1):
            out_lines.append(f"# Slide {idx}")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    out_lines.append(shape.text)
        return "\n".join(out_lines) + "\n"
    except Exception as e:
        return f"[error] reading PPTX {path.name}: {e}\n"


def _read_pdf(path: Path) -> str:
    try:
        fitz = importlib.import_module("fitz")
        doc = fitz.open(str(path))
        out = []
        for page in doc:
            out.append(page.get_text("text"))
        doc.close()
        return "\n".join(out)
    except Exception:
        pass
    try:
        high_level = importlib.import_module("pdfminer.high_level")
        return high_level.extract_text(str(path)) or ""
    except Exception as e:
        return f"[error] reading PDF {path.name}: {e}\n"


def _read_image(path: Path, ocr: bool) -> str:
    if not ocr:
        return f"[skip] {path.name}: OCR disabled.\n"
    global _pytesseract, _PIL_Image
    if _pytesseract is None:
        _pytesseract = _try_import("pytesseract")
    if _PIL_Image is None:
        try:
            from PIL import Image as _PIL

            _PIL_Image = _PIL
        except Exception:
            _PIL_Image = None
    if not (_pytesseract and _PIL_Image):
        return f"[missing] pytesseract/Pillow not installed for {path.name}\n"
    try:
        img = _PIL_Image.open(path)
        return _pytesseract.image_to_string(img)
    except Exception as e:
        return f"[error] OCR {path.name}: {e}\n"


def _read_json(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"[error] reading JSON {path.name}: {e}\n"


def _read_yaml(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = yaml.safe_load(f)
        return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    except Exception as e:
        return f"[error] reading YAML {path.name}: {e}\n"


def _read_xml(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[error] reading XML {path.name}: {e}\n"


def walk_and_extract(
    paths: Iterable[Path],
    ocr: bool = False,
    return_per_file: bool = False,
    base_dir: Path | None = None,
) -> Union[str, List[Tuple[Path, str]]]:
    """
    If return_per_file=False: returns one concatenated text blob with per-file headers.
    If return_per_file=True: returns a list of (file_path, text) pairs.
    base_dir: if provided, file_path is relative to this dir in the return list.
    """
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    files.append(child)
        elif p.is_file():
            files.append(p)

    seen = set()
    uniq: List[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            uniq.append(rp)
            seen.add(rp)

    def _rel(p: Path) -> Path:
        if base_dir:
            try:
                return p.resolve().relative_to(Path(base_dir).resolve())
            except Exception:
                pass
        return p

    if return_per_file:
        results: List[Tuple[Path, str]] = []
        for f in uniq:
            text = extract_text_from_path(f, ocr=ocr)
            if text and text.strip():
                results.append((_rel(f), text))
        return results

    out = io.StringIO()
    for f in uniq:
        out.write(f"\n\n===== FILE: {_rel(f).name} =====\n")
        out.write(extract_text_from_path(f, ocr=ocr))
    return out.getvalue()
