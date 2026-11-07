"""Build a small synthetic PDF corpus outside the repo for a wiring run.

Content is placeholder text about a fictional house, not taken from any real
document. Run:  uv run python scripts/make_probe_corpus.py /tmp/arbiter-probe
"""

from __future__ import annotations

import sys
from pathlib import Path


def make_pdf(path: Path, pages: list[str]) -> None:
    page_ids = [4 + i for i in range(len(pages))]
    content_ids = [4 + len(pages) + i for i in range(len(pages))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for _pid, cid in zip(page_ids, content_ids, strict=True):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>"
        )
    for page in pages:
        escaped = page.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 40 700 Td ({escaped}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    out = "%PDF-1.4\n"
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{obj}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n"
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/arbiter-probe")
    corpus = root / "korpus"
    corpus.mkdir(parents=True, exist_ok=True)

    make_pdf(
        corpus / "mietvertrag.pdf",
        [
            "Mietvertrag Haus Lindenweg 4. Die monatliche Miete betraegt 940 Euro "
            "zuzueglich 210 Euro Nebenkostenvorauszahlung.",
            "Die Kaution betraegt drei Nettokaltmieten und ist in drei Raten "
            "faellig. Schoenheitsreparaturen sind alle acht Jahre zu leisten.",
        ],
    )
    make_pdf(
        corpus / "nebenkosten.pdf",
        [
            "Nebenkostenabrechnung 2025. Heizung und Warmwasser 612 Euro, "
            "Muellgebuehr 148 Euro, Versicherung 96 Euro.",
            "Der Umlageschluessel folgt der Wohnflaeche. Nachzahlung 183 Euro, "
            "faellig innerhalb von 30 Tagen.",
        ],
    )
    make_pdf(
        corpus / "protokoll.pdf",
        [
            "Eigentuemerversammlung Protokoll. Beschluss: Das Dach des Hauses "
            "Lindenweg 4 wird im Sommer 2027 neu gedeckt.",
        ],
    )
    print(f"Korpus geschrieben: {corpus}")


if __name__ == "__main__":
    main()
