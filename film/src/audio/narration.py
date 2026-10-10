"""Synthesize the narration with Kokoro-82M and verify it with Whisper.

Outputs (out/audio/narration/):
  <id>.wav            24 kHz mono per segment
  narration.json      durations, per-word timestamps, ASR transcript + WER per segment
"""
import json, os, re, sys
import numpy as np
import soundfile as sf
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
from script.narration import SEGMENTS  # noqa: E402

OUT = os.path.join(ROOT, 'out', 'audio', 'narration')
SR = 24000
VOICE, SPEED = 'af_heart', 0.9


def synth(pipe, text):
    audio, words, t0 = [], [], 0.0
    for res in pipe(text, voice=VOICE, speed=SPEED, split_pattern=r'\n+'):
        a = res.audio.cpu().numpy() if hasattr(res.audio, 'cpu') else np.asarray(res.audio)
        for tk in (res.tokens or []):
            if tk.start_ts is not None and tk.text.strip() and re.search(r'\w', tk.text):
                words.append(dict(w=tk.text, t0=round(t0 + tk.start_ts, 3), t1=round(t0 + tk.end_ts, 3)))
        audio.append(a)
        t0 += len(a) / SR
    return np.concatenate(audio), words


def norm_words(s):
    s = re.sub(r'\[([^\]]+)\]\(/[^)]*/\)', r'\1', s)  # strip phoneme overrides
    s = s.lower().replace('-', ' ')
    return re.sub(r"[^a-z0-9' ]", ' ', s).split()


def wer(r, h):
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(len(r), 1)


NUM = {'55': 'fifty five', '87': 'eighty seven', '6.5': 'six and a half', '2019': 'twenty nineteen', '1.5': 'one and a half',
       '2.5': 'two and a half', '42': 'forty two', '1.3': 'one point three', '2017': 'twenty seventeen', "87's": "eighty seven's"}


def main(only=None):
    from kokoro import KPipeline
    from transformers import pipeline as hf_pipeline
    os.makedirs(OUT, exist_ok=True)
    tts = KPipeline(lang_code='a')
    asr = hf_pipeline('automatic-speech-recognition', model='openai/whisper-large-v3-turbo', device=0, dtype=torch.float16)
    meta = {}
    path = os.path.join(OUT, 'narration.json')
    if os.path.exists(path):
        meta = json.load(open(path))
    for sid, text, sub in SEGMENTS:
        if only and sid not in only:
            continue
        wav, words = synth(tts, text)
        # trim leading/trailing silence, keep 30 ms
        nz = np.where(np.abs(wav) > 0.005)[0]
        a0, a1 = max(nz[0] - int(0.03 * SR), 0), min(nz[-1] + int(0.03 * SR), len(wav))
        wav = wav[a0:a1]
        for w in words:
            w['t0'] = round(max(w['t0'] - a0 / SR, 0), 3); w['t1'] = round(max(w['t1'] - a0 / SR, 0), 3)
        sf.write(os.path.join(OUT, f'{sid}.wav'), wav.astype(np.float32), SR)
        hyp = asr({'raw': wav.astype(np.float32), 'sampling_rate': SR})['text'].strip()
        hw = ' '.join(norm_words(hyp))
        for k, v in NUM.items():
            hw = re.sub(rf'(?<![\w.]){re.escape(k)}(?![\w.])', v, hw)
        e = wer(norm_words(text), hw.split())
        meta[sid] = dict(duration=round(len(wav) / SR, 3), words=words, asr=hyp, wer=round(e, 4))
        print(f'{sid:7s} {len(wav)/SR:6.2f}s  WER {e:.3f}  | {hyp}')
    json.dump(meta, open(path, 'w'), indent=1)
    tot = sum(m['duration'] for m in meta.values())
    print(f'total narration {tot:.1f}s ({tot/60:.2f} min)')


if __name__ == '__main__':
    main(sys.argv[1:] or None)
