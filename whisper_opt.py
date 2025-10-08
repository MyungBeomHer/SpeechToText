import os, torch, re
import numpy as np
import librosa, soundfile as sf
import whisper  # openai-whisper
from datasets import load_dataset, Audio
import evaluate
from tqdm import tqdm

# ----------------------------
# 설정
# ----------------------------
MODEL_NAME = "medium"   # 다국어용 medium (medium.en 아님!)
LANG = "ko"
SPLIT = "test"          # "validation"으로 바꿔도 됩니다
MAX_SAMPLES = None      # 빠른 테스트면 예: 500 로 제한

# ----------------------------
# 텍스트 정규화 (한국어 WER/CER 안정화 목적)
#   - 대소문자 개념이 약하고, 특수문자/공백 차이로 점수 흔들림을 줄임
#   - 필요시 프로젝트 기준에 맞게 조정
# ----------------------------
_punct = re.compile(r"[^\u3131-\u318E\uAC00-\uD7A30-9a-zA-Z\s]")  # 한글/영문/숫자/공백만 허용
_multi_space = re.compile(r"\s+")

def normalize_korean(s: str) -> str:
    s = s.strip()
    s = _punct.sub(" ", s)          # 특수문자 제거 → 공백
    s = _multi_space.sub(" ", s)    # 다중 공백 → 단일 공백
    return s.strip().lower()

# ----------------------------
# 데이터셋 로드 (HF Datasets)
# audio 컬럼을 16kHz로 통일
# ----------------------------
ds = load_dataset("kresnik/zeroth_korean", split=SPLIT)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

if MAX_SAMPLES:
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

# ----------------------------
# 모델 로드
# GPU가 있으면 fp16 + cuda 사용, 없으면 자동으로 CPU
# ----------------------------
CUDA_VISIBLE_DEVICES=1
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_NAME, device=device)

# Whisper 디코딩 옵션 (필요시 조정)
decode_opts = dict(
    language=LANG,
    task="transcribe",
    beam_size=5,          # 좁히면 속도↑, 넓히면 품질↑
    best_of=5,
    temperature=0.0,      # 0.0이면 탐욕/빔서치 중심
    fp16=torch.cuda.is_available(),
)

# ----------------------------
# 추론 루프
# ----------------------------
pred_texts, ref_texts = [], []

for ex in tqdm(ds, total=len(ds)):
    # 16kHz 모노 numpy array 확보
    audio_np = ex["audio"]["array"]
    sr = ex["audio"]["sampling_rate"]
    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    # Whisper는 float32 ndarray도 입력 가능
    # (모델 내부에서 log-Mel로 변환)
    result = model.transcribe(audio_np, **decode_opts)
    hyp = result["text"]

    ref = ex["text"]  # Zeroth-Korean의 정답 텍스트 컬럼

    # 정규화(선택) – 평가 정책에 맞춰 on/off 가능
    hyp_norm = normalize_korean(hyp)
    ref_norm = normalize_korean(ref)

    pred_texts.append(hyp_norm)
    ref_texts.append(ref_norm)

# ----------------------------
# 평가 (WER/CER)
# ----------------------------
wer = evaluate.load("wer")
cer = evaluate.load("cer")

wer_score = wer.compute(references=ref_texts, predictions=pred_texts)  # 0.0 ~ 1.0
cer_score = cer.compute(references=ref_texts, predictions=pred_texts)  # 0.0 ~ 1.0

print(f"[Whisper {MODEL_NAME} @ {SPLIT}]")
print(f"Samples: {len(ref_texts)}")
print(f"WER: {wer_score*100:.2f}%")
print(f"CER: {cer_score*100:.2f}%")
print(f"Device: {device}")