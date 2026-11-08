"""Spec M: QESChunker, Kosten-Nutzen-Greedy mit Ueberlappung (Stufe 7.2)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from arbiter.chunk import ChunkParameter, QePaar, chunk_datei
from arbiter.chunk_qes import (
    beleg_indizes,
    chunke_qes,
    laengen_bereich,
    paare_im_fenster,
    treffer_indizes,
)

DOKUMENT = """\
Der Hund bellt im Hof.

Die Katze schlaeft auf dem Sofa.

Das Tor oeffnet um 17:00 und schliesst um 22:00.

Der Aufzug faesst acht Personen.

Die Terrasse ist im Sommer geoeffnet.
"""


class WortEmbedder:
    """Deterministischer Ersatz fuer den Embedder: zaehlt zwei Woerter."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vektoren = []
        for text in texts:
            woerter = text.lower().split()
            vektoren.append(
                [
                    float(sum(1 for w in woerter if "tor" in w)),
                    float(sum(1 for w in woerter if "hund" in w)),
                ]
            )
        return vektoren


def _paare() -> list[QePaar]:
    return [
        QePaar(
            frage="Wann oeffnet das Tor?",
            belege=("Das Tor oeffnet um 17:00 und schliesst um 22:00.",),
        ),
    ]


def test_beleg_indizes_verwirft_nicht_woertliche_belege() -> None:
    saetze = ["Der Hund bellt.", "Das Tor oeffnet um 17:00."]
    erlaubt = beleg_indizes([QePaar(frage="Wann?", belege=("Das Tor oeffnet um 17:00.",))], saetze)
    assert erlaubt == [(0, (1,))]
    verworfen = beleg_indizes([QePaar(frage="Wann?", belege=("Das Tor oeffnet bald.",))], saetze)
    assert verworfen == []
    fragment = beleg_indizes([QePaar(frage="Wann?", belege=("oeffnet um 17:00",))], saetze)
    assert fragment == [(0, (1,))]


def test_beleg_indizes_findet_mehrere_belege() -> None:
    saetze = ["Regel A gilt.", "Ausnahme B gilt."]
    erlaubt = beleg_indizes(
        [QePaar(frage="Gilt A?", belege=("Regel A gilt.", "Ausnahme B gilt."))], saetze
    )
    assert erlaubt == [(0, (0, 1))]


def test_treffer_indizes_nimmt_die_aehnlichsten() -> None:
    vektoren = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    assert treffer_indizes(vektoren, [1.0, 0.0], k=2) == {0, 2}
    assert treffer_indizes(vektoren, [0.0, 1.0], k=1) == {1}


def test_hit_gate_verlangt_treffer_und_beleg_im_fenster() -> None:
    belege = [(0, (2,))]
    treffer = [{0}]
    assert paare_im_fenster(0, 2, belege, treffer) == {(0, 2)}
    assert paare_im_fenster(2, 2, belege, treffer) == set(), "Beleg ohne Treffer zaehlt nicht"
    assert paare_im_fenster(0, 0, belege, treffer) == set(), "Treffer ohne Beleg zaehlt nicht"


def test_laengen_bereich_bleibt_im_rahmen() -> None:
    kurz, lang = laengen_bereich(DOKUMENT, max_chars=300)
    assert 0 < kurz < lang
    assert lang <= 300


def test_qes_bevorzugt_das_fenster_mit_beleg_und_treffer() -> None:
    chunks = chunke_qes(
        DOKUMENT,
        _paare(),
        parameter=ChunkParameter(strategie="qes", max_chars=200),
        embedder=WortEmbedder(),
    )
    assert chunks
    assert any("Das Tor oeffnet um 17:00" in chunk.text for chunk in chunks)
    assert all(chunk.strategie == "qes" for chunk in chunks)
    assert all(chunk.ende - chunk.start <= 200 for chunk in chunks), "K5 gilt auch hier"


def test_qes_laesst_ueberlappung_zu() -> None:
    # Drei Fragen in drei aufeinanderfolgenden Saetzen. Die Mindestlaenge eines Fensters
    # erzwingt zwei Saetze, also teilen sich die Fenster einen Satz.
    text = (
        "Der Hund bellt im Hof.\n\n"
        "Das Tor oeffnet um 17:00. Es schliesst um 22:00. Der Aufzug ist im Nordturm.\n\n"
        "Die Terrasse ist im Sommer geoeffnet.\n"
    )
    paare = [
        QePaar(frage="Wann oeffnet das Tor?", belege=("Das Tor oeffnet um 17:00.",)),
        QePaar(frage="Wann schliesst es?", belege=("Es schliesst um 22:00.",)),
        QePaar(frage="Wo ist der Aufzug?", belege=("Der Aufzug ist im Nordturm.",)),
    ]
    chunks = chunke_qes(
        text,
        paare,
        parameter=ChunkParameter(strategie="qes", max_chars=150),
        embedder=WortEmbedder(),
    )
    doppelt = [
        index
        for index in range(len(text))
        if sum(1 for chunk in chunks if chunk.start <= index < chunk.ende) > 1
    ]
    assert doppelt, "der Index darf sich ueberlappen"


def test_lange_absaetze_halten_das_fensterlimit_ein() -> None:
    # Lange Absaetze: das 90. Perzentil liegt ueber chunk_chars. Die Fensterlaenge darf
    # trotzdem nicht ueber das Limit wachsen (K5).
    text = "\n\n".join(
        " ".join(f"Absatz {absatz} Satz {satz} mit Inhalt." for satz in range(25))
        for absatz in range(5)
    )
    chunks = chunke_qes(
        text,
        [QePaar(frage="a?", belege=("Absatz 0 Satz 3 mit Inhalt.",))],
        parameter=ChunkParameter(strategie="qes", max_chars=200),
        embedder=WortEmbedder(),
    )
    assert chunks
    assert all(chunk.ende - chunk.start <= 200 for chunk in chunks)


def test_qes_ohne_verwertbare_belege_gibt_keine_chunks() -> None:
    chunks = chunke_qes(
        DOKUMENT,
        [QePaar(frage="Wann?", belege=("Dieser Satz steht nicht im Text.",))],
        parameter=ChunkParameter(strategie="qes", max_chars=200),
        embedder=WortEmbedder(),
    )
    assert chunks == []


def test_qes_ohne_embedder_gibt_keine_chunks() -> None:
    chunks = chunke_qes(
        DOKUMENT, _paare(), parameter=ChunkParameter(strategie="qes"), embedder=None
    )
    assert chunks == []


def test_chunk_datei_meldet_die_qes_strategie() -> None:
    ergebnis = chunk_datei(
        DOKUMENT,
        "handbuch.txt",
        ChunkParameter(strategie="qes", max_chars=200),
        qe_quelle=lambda text: _paare(),
        embedder=WortEmbedder(),
    )
    assert ergebnis.strategie == "qes"
    assert ergebnis.warnungen == []
    assert ergebnis.chunks


@given(st.text(alphabet="ab .1\n", min_size=1, max_size=400))
def test_property_kein_textverlust_bei_qes(text: str) -> None:
    saetze = [teil for teil in text.split(".") if teil.strip()]
    paare = [QePaar(frage="a?", belege=(saetze[0].strip() + ".",))] if saetze else []
    chunks = chunke_qes(
        text,
        paare,
        parameter=ChunkParameter(strategie="qes", max_chars=100),
        embedder=WortEmbedder(),
    )
    if not chunks:
        return
    abgedeckt = [False] * len(text)
    for chunk in chunks:
        assert chunk.ende - chunk.start <= 100, "K5: eigener Text bleibt im Limit"
        for index in range(chunk.start, chunk.ende):
            abgedeckt[index] = True
    for index, zeichen in enumerate(text):
        if not zeichen.isspace():
            assert abgedeckt[index], f"Zeichen {index} fehlt"


def test_nullvektor_bricht_die_trefferwahl_nicht() -> None:
    assert treffer_indizes([[0.0, 0.0], [1.0, 0.0]], [1.0, 0.0], k=1) == {1}
