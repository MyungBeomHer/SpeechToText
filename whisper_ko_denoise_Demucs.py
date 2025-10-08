import torch, re, evaluate
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from denoiser import pretrained
import numpy as np 

MODEL_ID = "seastar105/whisper-medium-ko-zeroth"
SPLIT = "test"
SR = 16000
BATCH = 32                  # GPU 여유에 맞게 조절(4~16 권장)
MAX_NEW_TOKENS = 224

import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model

# demucs = get_model("htdemucs").to("cuda" if torch.cuda.is_available() else "cpu").eval()

# @torch.no_grad()
# def enhance_batch_dns(wavs_np_list):  # 이름 재사용
#     outs = []
#     for x in wavs_np_list:
#         y = librosa.resample(x, orig_sr=SR, target_sr=32000)
#         ten = torch.from_numpy(y).float()[None, None, :].to(demucs.device)  # [1,1,T]
#         est = apply_model(demucs, ten, split=True, progress=False)[0].mean(0)  # [1,T] -> mono
#         est = est.squeeze(0).detach().cpu().numpy()
#         est = librosa.resample(est, orig_sr=32000, target_sr=SR)
#         outs.append(np.clip(est, -1.0, 1.0))
#     return outs

import numpy as np, torch, librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model

SR = 16000  # 이미 위쪽에 있다면 중복 정의 불필요

# 1) 전역에서 모델을 미리 로드 (NameError 방지)
_demucs = get_model("htdemucs").eval().to("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def enhance_batch_dns(wavs_np_list, sr=SR, target_sr=44100, mix=0.7):
    outs = []
    dev = next(_demucs.parameters()).device
    sources = getattr(_demucs, "sources", None)
    vocals_idx = sources.index("vocals") if sources and "vocals" in sources else None

    for x in wavs_np_list:
        y = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        ten_mono = torch.from_numpy(y).float()[None, None, :].to(dev)   # [1,1,T]
        ten = torch.cat([ten_mono, ten_mono], dim=1)                    # [1,2,T]

        est_all = apply_model(_demucs, ten, split=True, shifts=0, progress=False)

        # ---- 차원 정규화 ----
        if est_all.dim() == 4:            # [B,S,C,T]
            est_all = est_all.squeeze(0)  # -> [S,C,T]
        elif est_all.dim() == 3:
            pass                          # [S,C,T]
        elif est_all.dim() == 2:
            pass                          # [C,T]
        else:
            raise RuntimeError(f"Unexpected Demucs output shape: {tuple(est_all.shape)}")

        # ---- 소스/채널 선택 ----
        if est_all.dim() == 3:
            S, C, T = est_all.shape
            if vocals_idx is not None and vocals_idx < S:
                est = est_all[vocals_idx]     # [C,T]
            else:
                est = est_all.mean(0)         # [C,T]
            est = est.mean(0)                 # [T]
        else:  # [C,T]
            est = est_all.mean(0)             # [T]

        enh_16k = librosa.resample(est.detach().cpu().numpy(), orig_sr=target_sr, target_sr=sr)

        if mix < 1.0:
            L = min(len(enh_16k), len(x))
            enh_16k = mix * enh_16k[:L] + (1.0 - mix) * x[:L]

        outs.append(np.clip(enh_16k, -1.0, 1.0))
    return outs
# --- 정규화 ---
_punct = re.compile(r"[^\u3131-\u318E\uAC00-\uD7A30-9a-zA-Z\s]")
_multi_space = re.compile(r"\s+")
def normalize_ko(s):
    s = s.strip()
    s = _punct.sub(" ", s)
    s = _multi_space.sub(" ", s)
    return s.strip().lower()

# --- 데이터셋(16kHz 디코딩 유지) ---
ds = load_dataset("kresnik/zeroth_korean", split=SPLIT)
ds = ds.cast_column("audio", Audio(sampling_rate=SR))

# --- 모델/프로세서 ---
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, torch_dtype=dtype)

# 언어/태스크는 forced_decoder_ids로 지정
forced_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    chunk_length_s=30,
    torch_dtype=dtype,
    generate_kwargs={"forced_decoder_ids": forced_ids, "max_new_tokens": MAX_NEW_TOKENS},
)

# --- 배치 추론: KeyDataset을 사용하면 파이프라인이 자동으로 배치 처리 ---
preds, refs = [], []

# for i in tqdm(range(0, len(ds), BATCH)):
#     j_end = min(i + BATCH, len(ds))
#     rows = [ds[j] for j in range(i, j_end)]  # ← 각 행은 dict

#     audio_batch = [
#         {"array": r["audio"]["array"], "sampling_rate": SR}
#         for r in rows
#     ]
#     outs = asr(audio_batch, batch_size=BATCH)

#     for out, r in zip(outs, rows):
#         preds.append(normalize_ko(out["text"]))
#         refs.append(normalize_ko(r["text"]))
for i in tqdm(range(0, len(ds), BATCH)):
    j_end = min(i + BATCH, len(ds))
    rows = [ds[j] for j in range(i, j_end)]  # 각 행 dict

    # 1) 원본 wav 수집
    wavs = [r["audio"]["array"] for r in rows]        # list[np.ndarray], 16k mono
    # 2) SE 적용 (바로 이 한 줄)
    wavs = enhance_batch_dns(wavs)
    # 3) Whisper 배치 추론
    audio_batch = [{"array": w, "sampling_rate": SR} for w in wavs]
    outs = asr(audio_batch, batch_size=BATCH)

    for out, r in zip(outs, rows):
        preds.append(normalize_ko(out["text"]))
        refs.append(normalize_ko(r["text"]))

# --- 평가 ---
wer = evaluate.load("wer").compute(references=refs, predictions=preds)
cer = evaluate.load("cer").compute(references=refs, predictions=preds)
print(f"WER: {wer*100:.2f}%")
print(f"CER: {cer*100:.2f}%")