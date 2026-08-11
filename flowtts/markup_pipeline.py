"""
Markup layer on top of FlowTTSPipeline: parses tagged scripts like

    [WPM:60] [โทนเสียง:สดใส] [พักหายใจทุก 10 พยางค์]
    "เปิดแล้ว grand centre pointe pattaya....."

and renders them to a single wav file. F5-TTS itself has no notion of WPM,
tone, or breath pauses -- this module fakes all three by calling the
existing pipeline once per syllable-group chunk with a different `speed`
and reference voice, then stitching the pieces back together with
inserted silence.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from pythainlp.tokenize import syllable_tokenize

from flowtts.inference import FlowTTSPipeline

_TAG_RE = re.compile(r"\[([^\]]+)\]")
_BREATH_RE = re.compile(r"พักหายใจทุก\s*(\d+)\s*พยางค์")
_WPM_KEY = "WPM"
_TONE_KEY = "โทนเสียง"


def parse_markup_script(script: str) -> List[dict]:
    """Split a script into stanzas (tag line + quoted text line, separated by blank lines)."""
    blocks = []
    stanzas = re.split(r"\n\s*\n", script.strip())
    for stanza in stanzas:
        lines = [line for line in stanza.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        tag_line, text_line = lines[0], " ".join(lines[1:])

        wpm, tone, breath_every = None, None, None
        for tag in _TAG_RE.findall(tag_line):
            tag = tag.strip()
            breath_match = _BREATH_RE.search(tag)
            if breath_match:
                breath_every = int(breath_match.group(1))
                continue
            if ":" in tag:
                key, val = (part.strip() for part in tag.split(":", 1))
                if key.upper() == _WPM_KEY:
                    wpm = int(re.sub(r"[^\d]", "", val))
                elif key == _TONE_KEY:
                    tone = val

        text = text_line.strip().strip('"').strip("'")
        blocks.append({"wpm": wpm, "tone": tone, "breath_every": breath_every, "text": text})
    return blocks


def speed_from_wpm(wpm: Optional[int], base_wpm: float, min_speed: float = 0.5, max_speed: float = 2.0) -> float:
    """`speed` in FlowTTSPipeline is relative to the ref voice's own pace, not an absolute WPM,
    so we approximate: speed = requested_wpm / base_wpm (the WPM the ref clip is assumed to speak at)."""
    if wpm is None or not base_wpm:
        return 1.0
    return max(min_speed, min(max_speed, wpm / base_wpm))


def split_by_syllables(text: str, group_size: int) -> List[str]:
    """Group text into chunks of `group_size` syllables each (whitespace tokens ride along, not counted)."""
    tokens = syllable_tokenize(text, engine="dict")
    groups, buf, count = [], [], 0
    for tok in tokens:
        buf.append(tok)
        if tok.strip():
            count += 1
        if count >= group_size:
            groups.append("".join(buf).strip())
            buf, count = [], 0
    if buf:
        groups.append("".join(buf).strip())
    return groups or [text]


# Fill in real per-tone reference clips here. "default" is required as a fallback
# for blocks with no [โทนเสียง:...] tag or an unrecognized tone name.
TONE_PRESETS: Dict[str, dict] = {
    "default": {"ref_voice": "assets/000000.wav", "ref_text": None, "base_wpm": 60},
    "สดใส": {"ref_voice": "assets/ref_bright.wav", "ref_text": None, "base_wpm": 65},
    "ผู้ประกาศข่าว": {"ref_voice": "assets/ref_newscaster.wav", "ref_text": None, "base_wpm": 55},
}


def _get_preset(tone: Optional[str], tone_presets: Dict[str, dict]) -> dict:
    return tone_presets.get(tone, tone_presets["default"])


def _ensure_ref_text(pipeline: FlowTTSPipeline, preset: dict) -> str:
    """Transcribe the tone's ref voice once and cache it on the preset dict, so repeated
    chunks for the same tone don't re-run Whisper every time."""
    if not preset.get("ref_text"):
        preset["ref_text"] = pipeline.model.transcribe(preset["ref_voice"])
    return preset["ref_text"]


def synthesize_markup_script(
    pipeline: FlowTTSPipeline,
    script: str,
    output_file: str,
    tone_presets: Dict[str, dict] = TONE_PRESETS,
    breath_silence_ms: int = 300,
    block_gap_ms: int = 400,
) -> str:
    """Render a full markup script (multiple [WPM]/[โทนเสียง]/[พักหายใจ] blocks) to `output_file`."""
    blocks = parse_markup_script(script)
    if not blocks:
        raise ValueError("No parsable [tag] + \"text\" blocks found in script")

    chunk_dir = pipeline.temp_dir / "markup_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = None
    block_waves = []

    for block in blocks:
        preset = _get_preset(block["tone"], tone_presets)
        ref_text = _ensure_ref_text(pipeline, preset)
        speed = speed_from_wpm(block["wpm"], preset["base_wpm"])
        chunks = (
            split_by_syllables(block["text"], block["breath_every"])
            if block["breath_every"]
            else [block["text"]]
        )

        chunk_waves = []
        for chunk_text in chunks:
            if not chunk_text.strip():
                continue
            chunk_path = chunk_dir / f"{uuid.uuid4().hex}.wav"
            pipeline(
                text=chunk_text,
                ref_voice=preset["ref_voice"],
                ref_text=ref_text,
                output_file=str(chunk_path),
                speed=speed,
            )
            wav, sr = sf.read(str(chunk_path))
            sample_rate = sample_rate or sr
            chunk_waves.append(wav)
            os.remove(chunk_path)

        breath_gap = np.zeros(int(sample_rate * breath_silence_ms / 1000), dtype=chunk_waves[0].dtype)
        block_wave = chunk_waves[0]
        for wav in chunk_waves[1:]:
            block_wave = np.concatenate([block_wave, breath_gap, wav])
        block_waves.append(block_wave)

    block_gap = np.zeros(int(sample_rate * block_gap_ms / 1000), dtype=block_waves[0].dtype)
    final_wave = block_waves[0]
    for wav in block_waves[1:]:
        final_wave = np.concatenate([final_wave, block_gap, wav])

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, final_wave, sample_rate)
    return output_file


if __name__ == "__main__":
    from flowtts.inference import ModelConfig, AudioConfig

    SCRIPT = """
[WPM:60] [โทนเสียง:สดใส] [พักหายใจทุก 10 พยางค์]
"เปิดแล้ว grand centre pointe pattaya"

[WPM:70] [โทนเสียง:ผู้ประกาศข่าว] [พักหายใจทุก 14 พยางค์]
"มองเห็นทัศนียภาพ อันงดงามของหาดพัทยา"
"""

    model_config = ModelConfig(
        language="th",
        checkpoint="hf://biodatlab/ThonburianTTS/megaF5/mega_f5_last.safetensors",
        vocab_file="hf://biodatlab/ThonburianTTS/megaF5/mega_vocab.txt",
    )
    pipeline = FlowTTSPipeline(model_config=model_config, audio_config=AudioConfig())
    out = synthesize_markup_script(pipeline, SCRIPT, "outputs_markup/demo.wav")
    print(f"Saved: {out}")
