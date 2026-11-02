# -*- coding: utf-8 -*-
"""Sinh anh PNG cho tung so do TikZ trong docs/so_do/.

Cach dung:
    python build_png.py

Quy trinh:
    1. Sinh wrapper (article + preamble cua tai lieu chinh, dat caption dung so).
    2. xelatex  ->  PDF trong _build/
    3. Render PDF -> PNG, cat sat noi dung (pymupdf); neu khong co pymupdf
       thi fallback sang pdftoppm (toan trang A4).

Yeu cau: xelatex (MiKTeX/TeX Live) trong PATH; pymupdf hoac pdftoppm (poppler).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SODO = Path(__file__).resolve().parent
DOCS = SODO.parent
MAIN = DOCS / "kien_truc_he_thong.tex"
BUILD = SODO / "_build"
PNG = SODO / "png"
DPI = 200
PADDING = 10  # point, gia tri an toan quanh noi dung sau khi cat


def load_preamble() -> str:
    """Chup phan preamble cua tai lieu chinh (tu \\documentclass den \\begin{document})."""
    text = MAIN.read_text(encoding="utf-8")
    start = text.index("\\documentclass")
    end = text.index("\\begin{document}")
    return text[start:end]


def run(cmd: list[str], cwd: Path) -> None:
    r = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        sys.stdout.write((r.stdout or "")[-4000:])
        sys.stderr.write((r.stderr or "")[-4000:])
        raise SystemExit("LOI khi chay: " + " ".join(cmd))


def content_rect(page) -> "object":
    """Hop bao quanh toan bo chu + do hoa + anh tren trang (toan tu pymupdf)."""
    import pymupdf

    rect = pymupdf.Rect()
    for block in page.get_text("blocks"):
        rect |= pymupdf.Rect(block[:4])
    for drawing in page.get_drawings():
        if drawing.get("rect"):
            rect |= drawing["rect"]
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            rect |= r
    if rect.is_empty or not rect.is_valid:
        rect = page.rect
    # + padding, khong vuot qua trang
    rect = pymupdf.Rect(
        max(page.rect.x0, rect.x0 - PADDING),
        max(page.rect.y0, rect.y0 - PADDING),
        min(page.rect.x1, rect.x1 + PADDING),
        min(page.rect.y1, rect.y1 + PADDING),
    )
    return rect


def render_png(pdf: Path, out: Path) -> None:
    """Render PDF (1 trang) -> PNG, cat sat noi dung neu co pymupdf."""
    try:
        import pymupdf
    except ImportError:
        run(["pdftoppm", "-png", "-r", str(DPI), "-singlefile",
             str(pdf), str(out.with_suffix(""))], cwd=SODO)
        return

    doc = pymupdf.open(pdf)
    page = doc[0]
    pix = page.get_pixmap(clip=content_rect(page), dpi=DPI)
    pix.save(out)
    doc.close()


def figure_order() -> list[Path]:
    """Thu tu so do dung trong tai lieu chinh (theo cac \\input{so_do/...}).

    Doc tu file .tex chinh de so caption trong PNG luon trung so voi tai lieu,
    ke ca khi ta chen/doi thu tu so do.
    """
    import re

    text = MAIN.read_text(encoding="utf-8")
    names = re.findall(r"\\input\{(so_do/[^}]+)\}", text)
    seen: list[Path] = []
    for name in names:
        path = DOCS / (name + ".tex")
        if path.exists() and path not in seen:
            seen.append(path)
    return seen


def main() -> None:
    figs = figure_order()
    if not figs:
        raise SystemExit("khong tim thay \\input{so_do/...} nao trong " + str(MAIN))

    BUILD.mkdir(exist_ok=True)
    PNG.mkdir(exist_ok=True)
    preamble = load_preamble()

    for i, fig in enumerate(figs, start=1):
        name = fig.stem
        # dat ten `wrap_...` + \input duong dan tuyet doi de khong bi trung ten
        # voi file so do goc khi TeX tim kiem tu thu muc chua file chinh.
        wrapper = BUILD / ("wrap_" + name + ".tex")
        # dat counter = i-1 de caption trong PNG cung so voi tai lieu chinh
        wrapper.write_text(
            "% Tu dong sinh boi build_png.py - khong sua tay.\n"
            + preamble
            + "\\setcounter{figure}{%d}\n" % (i - 1)
            + "\\begin{document}\n"
            + "\\pagestyle{empty}\n"      # an so trang o day, de bbox sat so do
            + "\\thispagestyle{empty}\n"
            + "\\input{%s}\n" % fig.as_posix()
            + "\\end{document}\n",
            encoding="utf-8",
        )
        run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory=" + BUILD.name,
             (Path(BUILD.name) / wrapper.name).as_posix()],
            cwd=SODO,
        )
        render_png(BUILD / ("wrap_" + name + ".pdf"), PNG / (name + ".png"))
        print("  [ok] png/%s.png" % name)

    print("Xong ->", PNG)


if __name__ == "__main__":
    main()
