"""Spec M3: Frage-Beleg-Paare erzeugen und pruefen (Stufe 7.2)."""

from __future__ import annotations

from arbiter.chunk_qe import erzeuge_qe, mache_qe_quelle

TEXT = """\
Das untere Tor oeffnet um 17:00.
Es schliesst um 22:00.
Der Aufzug faesst acht Personen.
Die Terrasse ist im Sommer geoeffnet.
"""


def _chat(antwort: str) -> object:
    def ruf(system: str, user: str) -> str:
        assert "FRAGE:" in user, "die Anweisung nennt das Format"
        return antwort

    return ruf


def test_parst_frage_und_beleg() -> None:
    antwort = "FRAGE: Wann oeffnet das Tor?\nBELEG: Das untere Tor oeffnet um 17:00.\n"
    paare = erzeuge_qe(TEXT, chat=_chat(antwort))  # type: ignore[arg-type]
    assert paare[0].frage == "Wann oeffnet das Tor?"
    assert paare[0].belege == ("Das untere Tor oeffnet um 17:00.",)


def test_verwirft_belege_die_nicht_im_text_stehen() -> None:
    antwort = "FRAGE: Wann?\nBELEG: Das Tor oeffnet um 17:00 Uhr.\n"
    paare = erzeuge_qe(TEXT, chat=_chat(antwort))  # type: ignore[arg-type]
    assert paare == []


def test_verwirft_zu_kurze_fragmente() -> None:
    antwort = "FRAGE: Wann?\nBELEG: Tor\n"
    assert erzeuge_qe(TEXT, chat=_chat(antwort)) == []  # type: ignore[arg-type]


def test_verwirft_mehrdeutige_fragmente() -> None:
    text = "Das Tor oeffnet um 17:00.\nDas Tor oeffnet um 22:00.\n"
    antwort = "FRAGE: Wann?\nBELEG: Das Tor oeffnet\n"
    assert erzeuge_qe(text, chat=_chat(antwort)) == []  # type: ignore[arg-type]


def test_nimmt_woertliches_fragment_an() -> None:
    # Gemessen 2026-11-08 im echten Lauf: das Modell zitiert Teilsaetze, aber woertlich.
    antwort = "FRAGE: Wann?\nBELEG: untere Tor oeffnet um 17\n"
    paare = erzeuge_qe(TEXT, chat=_chat(antwort))  # type: ignore[arg-type]
    assert len(paare) == 1


def test_mehrere_belege_gehoeren_zu_einer_frage() -> None:
    antwort = (
        "FRAGE: Wann oeffnet und schliesst das Tor?\n"
        "BELEG: Das untere Tor oeffnet um 17:00.\n"
        "BELEG: Es schliesst um 22:00.\n"
    )
    paare = erzeuge_qe(TEXT, chat=_chat(antwort))  # type: ignore[arg-type]
    assert len(paare) == 1
    assert len(paare[0].belege) == 2


def test_haelt_die_obergrenze_ein() -> None:
    antwort = "".join(
        f"FRAGE: Frage {nummer}?\nBELEG: Das untere Tor oeffnet um 17:00.\n" for nummer in range(5)
    )
    paare = erzeuge_qe(TEXT, chat=_chat(antwort), max_fragen=2)  # type: ignore[arg-type]
    assert len(paare) == 2


def test_leere_antwort_ergibt_keine_paare() -> None:
    assert erzeuge_qe(TEXT, chat=_chat("")) == []  # type: ignore[arg-type]


def test_modellfehler_ergibt_keine_paare() -> None:
    def kaputt(system: str, user: str) -> str:
        raise RuntimeError("nicht erreichbar")

    assert erzeuge_qe(TEXT, chat=kaputt) == []


def test_fragen_werden_ueber_mehrere_fenster_verteilt() -> None:
    gesehen: list[str] = []

    def zaehlend(system: str, user: str) -> str:
        gesehen.append(user)
        return "FRAGE: Wann?\nBELEG: Das untere Tor oeffnet um 17:00.\n"

    text = "Das untere Tor oeffnet um 17:00. " * 120  # mehr als ein Fenster
    erzeuge_qe(text, chat=zaehlend, max_fragen=4, fenster=500, fenster_max=2)
    assert len(gesehen) == 2, "hoechstens fenster_max Aufrufe"


def test_doppelte_fragen_zaehlen_einmal() -> None:
    antwort = (
        "FRAGE: Wann?\nBELEG: Das untere Tor oeffnet um 17:00.\n"
        "FRAGE: Wann?\nBELEG: Es schliesst um 22:00.\n"
    )
    paare = erzeuge_qe(TEXT, chat=_chat(antwort))  # type: ignore[arg-type]
    assert len(paare) == 1


def test_quelle_reicht_grenzen_und_modell_weiter() -> None:
    def chat(system: str, user: str) -> str:
        return "FRAGE: Wann?\nBELEG: Das untere Tor oeffnet um 17:00.\n"

    quelle = mache_qe_quelle(base_url="http://test", model="m", chat=chat, max_fragen=1)
    assert len(quelle(TEXT)) == 1
