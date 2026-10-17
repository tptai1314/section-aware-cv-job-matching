"""Tạo PDF văn bản thuần (text layer) tối giản để test analyze-cv.

Cách dùng: python scripts/make_fixture.py <output.pdf>
"""

from __future__ import annotations

import sys

TEXT = (
    "Software Engineer with 4 years of experience building React apps. "
    "Skills: React, Node.js, TypeScript, PostgreSQL. "
    "Worked at Acme Corp as Frontend Developer from Mar 2021 to Present "
    "building the design system. BSc Computer Science at HCMUT, 2016-2020. "
    "Project Shop App using React Native and Stripe."
)


def build_pdf(text: str) -> bytes:
    out = bytearray(b"%PDF-1.4\n")

    def add_obj(num: int, content: bytes) -> int:
        offset = len(out)
        out.extend(f"{num} 0 obj\n".encode("latin-1"))
        out.extend(content)
        out.extend(b"\nendobj\n")
        return offset

    offsets: dict[int, int] = {}
    offsets[1] = add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    offsets[2] = add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    offsets[3] = add_obj(
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
    )
    stream = b"BT /F1 12 Tf 72 720 Td (" + text.encode("latin-1") + b") Tj ET\n"
    offsets[4] = add_obj(
        4,
        f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"endstream",
    )
    offsets[5] = add_obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_pos = len(out)
    out.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for i in range(1, 6):
        out.extend(f"{offsets[i]:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return bytes(out)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cv_fixture.pdf"
    with open(out, "wb") as fh:
        fh.write(build_pdf(TEXT))
    print(f"wrote {out}")