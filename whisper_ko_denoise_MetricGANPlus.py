import torch, re, evaluate
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from denoiser import pretrained
import numpy as np 
from speechbrain.pretrained import SpectralMaskEnhancement

MODEL_ID = "seastar105/whisper-medium-ko-zeroth"
SPLIT = "test"
SR = 16000
BATCH = 32                  # GPU 여유에 맞게 조절(4~16 권장)
MAX_NEW_TOKENS = 224

# sb_enh = SpectralMaskEnhancement.from_hparams(
#     source="speechbrain/metricgan-plus-voicebank",
#     savedir="pretrained_models/metricganplus",
# ).to("cuda" if torch.cuda.is_available() else "cpu")
enh_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
sb_enh = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricganplus",
    run_opts={"device": enh_device_str},   # ★ 내부 모듈/전처리까지 해당 device로 고정
)

@torch.no_grad()
def enhance_batch_dns(wavs_np_list):  # list of np.ndarray (16k mono)
    # [B, T] float32
    wavs = [torch.from_numpy(x).float() for x in wavs_np_list]
    L = max(w.shape[0] for w in wavs)
    batch = torch.stack([torch.nn.functional.pad(w, (0, L - w.shape[0])) for w in wavs], dim=0)  # [B, T]

    # 각 샘플 상대 길이(0~1), device 일치
    lens = torch.tensor([w.shape[0] / L for w in wavs], dtype=torch.float32)

    # ★ 모델의 device와 동일하게 이동
    device = sb_enh.device  # speechbrain Inference 클래스가 보유
    batch = batch.to(device)
    lens  = lens.to(device)

    # 전용 API 호출 (forward 아님)
    enhanced = sb_enh.enhance_batch(batch, lens)  # [B, T] on device

    # 원 길이로 자르고 numpy 반환
    outs = [enhanced[i, :wavs[i].shape[0]].detach().cpu().numpy() for i in range(len(wavs))]
    return [np.clip(o, -1.0, 1.0) for o in outs]

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