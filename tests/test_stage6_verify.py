import io
import wave

from pipeline.stage6_tts import _flag_delivery, _respell


def _params():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00")
    buf.seek(0)
    with wave.open(buf, "rb") as w:
        return w.getparams()


def test_flag_delivery_catches_rate_anomaly():
    # 10 seconds of audio for a 3-word sentence: a laughing fit, not speech
    body = b"\x00\x00" * (24000 * 10)
    problem = _flag_delivery(0, _params(), body, "three word line")
    assert problem and "rate" in problem


def test_flag_delivery_degrades_without_transcribe():
    # normal-rate clip, transcribe endpoint unreachable (port 1) -> no flag
    body = b"\x00\x00" * 24000  # 1s for 3 words
    assert _flag_delivery(1, _params(), body, "three word line") is None


def test_respell_only_touches_mapped_words():
    assert _respell("House Fiyero took Hala", {"Fiyero": "Fiero"}) == \
        "House Fiero took Hala"
    assert _respell("no map here", None) == "no map here"
