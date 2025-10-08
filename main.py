#CUDA_VISIBLE_DEVICES=3 python main_FRU.py
import os, re, torch, numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
import torch

from datasets import load_dataset, Audio
import evaluate

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    IntervalStrategy
)
import types
import torch.nn as nn


import soundfile as sf
import librosa

# --------------------------
# 설정
# --------------------------
# MODEL_ID = "openai/whisper-medium"          # 다국어 medium (medium.en 아님!)
MODEL_ID = "seastar105/whisper-medium-ko-zeroth"
DATASET_ID = "kresnik/zeroth_korean"
SAMPLE_RATE = 16000
LANGUAGE = "korean"                          # processor용 언어 프롬프트
TASK = "transcribe"

# 학습 하이퍼파라미터(리소스에 맞게 조정)
NUM_EPOCHS = 5
LR = 5e-6                                  # 디코더만 FT면 1e-4~5e-5, 전체 FT면 1e-5 권장
TRAIN_BS = 8
EVAL_BS  = 8
GRAD_ACC = 2
WARMUP_RATIO = 0.05

USE_FP16 = torch.cuda.is_available()
FREEZE_ENCODER = True                        # 안전한/가벼운 설정(권장). False면 전체 FT.
# USE_LORA = False                           # 필요하면 LoRA로 전환 (아래 주석 참조)

SEED = 42

# --------------------------
# 텍스트 정규화 (WER/CER 안정화)
# --------------------------
_punct = re.compile(r"[^\u3131-\u318E\uAC00-\uD7A30-9a-zA-Z\s]")
_multi = re.compile(r"\s+")
def normalize_ko(s: str) -> str:
    s = s.strip()
    s = _punct.sub(" ", s)
    s = _multi.sub(" ", s)
    return s.strip().lower()

# --------------------------
# 데이터셋 로드
# --------------------------
train_ds = load_dataset(DATASET_ID, split="train")
test_ds  = load_dataset(DATASET_ID, split="test")

# 오디오 16kHz로 디코딩
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
test_ds  = test_ds.cast_column("audio",  Audio(sampling_rate=SAMPLE_RATE))
# train_ds = train_ds.cast_column("audio", Audio(decode=False))
# test_ds  = test_ds.cast_column("audio",  Audio(decode=False))

# --------------------------
# 모델/프로세서 준비
# --------------------------
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID,use_safetensors=True)

target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

from peft import LoraConfig, get_peft_model

OUTPUT_DIR = "./whisper-LORA/Encoder=FRU-Adapter+LORA_Decoder=LORA"

peft_cfg = LoraConfig(
    r=128,                 
    lora_alpha=64,       
    lora_dropout=0.05,    
    bias="none",
    target_modules=target_modules,
)
# rank (8~32) hidden_dim / # scaling 벡터에 곱해지는 가중치 
model = get_peft_model(model, peft_cfg)
# (선택) 모든 파라미터 끄고 LoRA만 켜기 — 가장 확실
for p in model.parameters():
    p.requires_grad_(False)
for n,p in model.named_parameters():
    if "lora_" in n:  # PEFT가 붙인 LoRA 파라미터 이름
        p.requires_grad_(True)

def print_trainable(m):
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M  ({100*trainable/total:.2f}%)")
print_trainable(model)    

#----------------------
# 언어/태스크 프롬프트 강제
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
model.config.suppress_tokens = []  # (선택) 한국어 평가에서 불필요 토큰 억제 완화/조정 가능
model.config.use_cache = False     # gradient checkpointing과 충돌 방지용
#------------------------

# --------------------------
# 전처리 함수
# --------------------------

def prepare_batch(batch):
    # 1) 오디오 → 입력 피처
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # 2) 라벨 토큰화 (정규화 후) — as_target_tokenizer() 제거
    text = normalize_ko(batch["text"])
    labels = processor.tokenizer(text, add_special_tokens=False)
    batch["labels"] = labels["input_ids"]
    return batch

train_ds = train_ds.map(prepare_batch, remove_columns=train_ds.column_names, num_proc=4)
test_proc = test_ds.map(prepare_batch,  remove_columns=test_ds.column_names,  num_proc=4)

# --------------------------
# Collator
# --------------------------
@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        # input_features pad
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # labels pad -> -100 마스킹
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch

collator = DataCollatorSpeechSeq2Seq(processor=processor)

# --------------------------
# Metric 함수 (WER/CER)
# --------------------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    # pred.predictions: (batch, seq_len) int ids (generate 사용 시)
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # -100 복원
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 텍스트 디코드
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 정규화 동일 적용
    pred_norm  = [normalize_ko(s) for s in pred_str]
    label_norm = [normalize_ko(s) for s in label_str]

    wer = wer_metric.compute(references=label_norm, predictions=pred_norm)
    cer = cer_metric.compute(references=label_norm, predictions=pred_norm)
    return {"wer": wer, "cer": cer}

# --------------------------
# 학습 설정
# --------------------------
gen_kwargs = dict(max_new_tokens=224, num_beams=1)  # 한국어: greedy/beam1가 보통 안정적

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=NUM_EPOCHS,
    fp16=USE_FP16,
    eval_strategy=IntervalStrategy.STEPS,
    predict_with_generate=True,
    generation_max_length=224,          # v5 경고 회피: max_new_tokens 사용 None
    generation_num_beams=1, #
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=2,
    ddp_find_unused_parameters=True,  # ← DDP가 '안쓴 파라미터'를 허용
    gradient_checkpointing=False,
    dataloader_num_workers=4,
    report_to="none",
    remove_unused_columns=True, # False
    metric_for_best_model="wer",
    label_names=["labels"],
    seed=SEED,
)
# evaluation_strategy="steps",
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_ds,
#     eval_dataset=test_proc,
#     data_collator=collator,
#     tokenizer=processor.tokenizer,
#     compute_metrics=compute_metrics,
#     preprocess_logits_for_metrics=None,
# )
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_proc,
    data_collator=collator,
    processing_class=processor,     # ← 여기!
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=None,
)

# tokenizer=processor.tokenizer,  # ← 지우기
model.main_input_name = "input_features"
# generate 설정 주입
trainer.model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids
# trainer.model.generation_config.num_beams = gen_kwargs["num_beams"]
# trainer.model.generation_config.max_new_tokens = gen_kwargs["max_new_tokens"]

# --------------------------
# 학습
# --------------------------
trainer.train()

# --------------------------
# validation 최종 점수 확인
# --------------------------
val_metrics = trainer.evaluate()
print({k: (v*100 if k in ["wer","cer"] else v) for k, v in val_metrics.items() if k in ["eval_wer","eval_cer","eval_loss"]})

# --------------------------
# test 평가
# --------------------------
test_out = trainer.predict(test_proc, metric_key_prefix="test")
test_metrics = {k: (v*100 if k in ["test_wer","test_cer"] else v)
                for k,v in test_out.metrics.items() if k in ["test_wer","test_cer","test_loss"]}
print(test_metrics)

# 모델/프로세서 저장
# trainer.save_model(OUTPUT_DIR)
# processor.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)  # 어댑터(LoRA) 가중치만 저장

# if you want load LORA, use this code
# from peft import PeftModel
# base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
# lora = PeftModel.from_pretrained(base, OUTPUT_DIR)
# merged = model.merge_and_unload()  # LoRA → 본체에 합치기
# merged.save_pretrained(OUTPUT_DIR + "-merged")