"""TUI zum Ausprobieren: Frage stellen, Antwort, Quellen, Warnungen, Laufzeit.

Deutsch, eine Ansicht, eine Tastaturbelegung:

    Enter    Frage abschicken
    Ctrl+S   stattdessen suchen (Passagen mit Reranker-Score, ohne Antwort)
    Ctrl+I   Indexieren (fragt vorher nach, weil es je nach OCR lange dauert)
    Ctrl+L   Verlauf leeren
    Ctrl+Q   beenden

Die Rechenarbeit kommt aus :class:`arbiter.pipeline.Pipeline`; diese Datei enthält
nur Oberfläche.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Label, Static

from arbiter.pipeline import Antwort, Pipeline, SearchResult


@dataclass(frozen=True)
class Zug:
    """One question with everything the screen needs to show about it."""

    frage: str
    antwort: Antwort | None = None
    treffer: SearchResult | None = None
    dauer: float = 0.0
    fehler: str | None = None


def quellen_text(antwort: Antwort) -> str:
    """One line per source: file, page, first characters of the passage."""
    if not antwort.quellen:
        return "keine Quelle"
    zeilen = []
    for quelle in antwort.quellen:
        anfang = " ".join(quelle.text.split())[:110]
        zeilen.append(f"  • {quelle.quelle}, S. {quelle.seite} — {anfang}…")
    return "\n".join(zeilen)


def treffer_text(treffer: SearchResult) -> str:
    """The passages of a search with their scores, for the search mode."""
    if not treffer.passages:
        return "keine Passage gefunden"
    zeilen = []
    for passage in treffer.passages:
        score = "—" if passage.rerank_score is None else f"{passage.rerank_score:+.2f}"
        anfang = " ".join(passage.text.split())[:90]
        zeilen.append(
            f"  {passage.rrf_rank:>2}. [{score}] {passage.quelle}, S. {passage.seite} — {anfang}…"
        )
    return "\n".join(zeilen)


class ArbiterTui(App[None]):
    """Frage oben, Antwort darunter, Verlauf darunter, Status unten."""

    CSS = """
    Screen { layout: vertical; }
    #eingabe { dock: top; height: 3; }
    #stand { height: 1; color: $text-muted; padding: 0 1; }
    #verlauf { height: 1fr; border: round $primary; padding: 0 1; }
    .zug { margin: 0 0 1 0; }
    .frage { text-style: bold; }
    .antwort { }
    .quellen { color: $text-muted; }
    .warnung { color: $warning; }
    .fehler { color: $error; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+q", "quit", "Beenden"),
        Binding("ctrl+s", "suche", "Suchen"),
        Binding("ctrl+i", "indexieren", "Indexieren"),
        Binding("ctrl+l", "leeren", "Verlauf leeren"),
    ]

    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.suchmodus = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(
            placeholder="Frage an den Korpus …  (Enter = fragen, Strg+S = suchen)", id="eingabe"
        )
        yield Static("", id="stand")
        with VerticalScroll(id="verlauf"):
            yield Vertical(id="zuege")
        yield Footer()

    def on_mount(self) -> None:
        self._stand()
        self.query_one("#eingabe", Input).focus()

    def _stand(self) -> None:
        rag = self.pipeline.rag
        try:
            zeilen = self.pipeline.store.count(rag.name)
        except Exception:
            zeilen = -1
        modus = "Suche" if self.suchmodus else "Antwort"
        self.query_one("#stand", Static).update(
            f"RAG {rag.name} · {zeilen} Abschnitte · Modus {modus} · "
            f"Reranker {'an' if rag.rerank else 'aus'} · "
            f"Gate {rag.gate_prompt} über {rag.gate_kandidaten} Passagen"
        )

    @on(Input.Submitted, "#eingabe")
    def _abgeschickt(self, event: Input.Submitted) -> None:
        frage = event.value.strip()
        if not frage:
            return
        self.query_one("#eingabe", Input).value = ""
        if self.suchmodus:
            self._suche(frage)
        else:
            self._frage(frage)

    def _frage(self, frage: str) -> None:
        laeuft = Label("…", classes="antwort")
        kasten = Vertical(Label(f">> {frage}", classes="frage"), laeuft, classes="zug")
        self.query_one("#zuege", Vertical).mount(kasten)
        self.query_one("#verlauf", VerticalScroll).scroll_end(animate=False)
        self._frage_im_hintergrund(frage, laeuft)

    @work(thread=True)
    def _frage_im_hintergrund(self, frage: str, laeuft: Label) -> None:
        start = time.monotonic()
        try:
            antwort = self.pipeline.ask(frage)
        except Exception as exc:
            self.call_from_thread(laeuft.update, f"Fehler: {exc}")
            self.call_from_thread(laeuft.add_class, "fehler")
            return
        dauer = time.monotonic() - start
        self.call_from_thread(self._zeige_antwort, laeuft, antwort, dauer)

    def _zeige_antwort(self, laeuft: Label, antwort: Antwort, dauer: float) -> None:
        teile: list[str] = []
        if antwort.text is None:
            teile.append(f"keine Antwort ({antwort.grund})")
        else:
            teile.append(antwort.text)
            teile.append(quellen_text(antwort))
        teile.append(f"[{dauer:.1f}s]")
        for warnung in antwort.warnungen[:3]:
            teile.append(f"⚠ {warnung}")
        laeuft.update("\n".join(teile))
        laeuft.add_class("warnung" if antwort.text is None else "antwort")
        self.query_one("#verlauf", VerticalScroll).scroll_end(animate=False)

    def _suche(self, frage: str) -> None:
        laeuft = Label("…", classes="antwort")
        kasten = Vertical(Label(f">> {frage}", classes="frage"), laeuft, classes="zug")
        self.query_one("#zuege", Vertical).mount(kasten)
        self.query_one("#verlauf", VerticalScroll).scroll_end(animate=False)
        self._suche_im_hintergrund(frage, laeuft)

    @work(thread=True)
    def _suche_im_hintergrund(self, frage: str, laeuft: Label) -> None:
        start = time.monotonic()
        try:
            treffer = self.pipeline.search(frage)
        except Exception as exc:
            self.call_from_thread(laeuft.update, f"Fehler: {exc}")
            return
        dauer = time.monotonic() - start
        self.call_from_thread(laeuft.update, f"{treffer_text(treffer)}\n[{dauer:.1f}s]")
        self.call_from_thread(self.query_one("#verlauf", VerticalScroll).scroll_end, False)

    def action_suche(self) -> None:
        self.suchmodus = not self.suchmodus
        self._stand()
        self.notify(f"Modus: {'Suche' if self.suchmodus else 'Antwort'}")

    def action_leeren(self) -> None:
        self.query_one("#zuege", Vertical).remove_children()
        self.notify("Verlauf geleert")

    def action_indexieren(self) -> None:
        self.notify("Index läuft …", timeout=3)
        self._indexieren_im_hintergrund()

    @work(thread=True)
    def _indexieren_im_hintergrund(self) -> None:
        try:
            bericht = self.pipeline.index()
        except Exception as exc:
            self.call_from_thread(self.notify, f"Index fehlgeschlagen: {exc}", severity="error")
            return
        self.call_from_thread(
            self.notify,
            f"{bericht.files} gelesen, {bericht.unchanged} übersprungen, "
            f"{bericht.changed} neu, {bericht.ocr_pages} Seiten OCR",
            severity="information",
            timeout=6,
        )
        self.call_from_thread(self._stand)


def main() -> None:
    """Startet die Oberfläche; fehlende Variablen kommen aus ``arbiter.env``."""
    from pathlib import Path

    from arbiter.config import load_config
    from arbiter.runtime import lade_umgebung

    umgebung = lade_umgebung()
    cfg = load_config(Path(umgebung.get("ARBITER_CONFIG", "arbiter.yaml")), umgebung)
    name = umgebung.get("ARBITER_RAG") or next(iter(cfg.rags))
    ArbiterTui(Pipeline(cfg, cfg.rags[name])).run()


if __name__ == "__main__":
    main()
