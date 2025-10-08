## AI based Speech-To-Text

팀원 : [허명범](https://github.com/MyungBeomHer)

### 프로젝트 주제 
음성 인식 모델 개발

### 프로젝트 언어 및 환경
프로젝트 언어 : Pytorch

### Dataset
- [Zeroth-korean Dataset](https://huggingface.co/datasets/kresnik/zeroth_korean?utm_source=chatgpt.com)

### Requirements
```bash
pip install -r requirements.txt
```

### 1st train
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 main.py
```

### next train (you already downloaded the dataset and safetensors of model. So you can not download them one more time.)
```bash
export HF_HOME=/data/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 main.py
```

### Model
<p align="center">
  <img src="/figure/model.png" width=100%> <br>
</p>

```
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
model = get_peft_model(model, peft_cfg)
for p in model.parameters():
    p.requires_grad_(False)
for n,p in model.named_parameters():
    if "lora_" in n:  
        p.requires_grad_(True)
```
[main.py](main.py)

- Benchmark (Zeroth-Korean Dataset)

|Model|Denoiser|Trainable Params|WER|
|:------:|:------:|:---:|:---:|
|Whisper|X|769M|3.64|
|Whisper|Facebook-denoiser|769M|4.40||
|Whisper|MetricGAN+|769M|23.87||
|Whisper|DemucsV4|769M|3.72||
|**Whisper+LORA(ours)**|X|75M|3.62||

### Reference Repo
- [Whisper](https://github.com/openai/whisper)
- [Whisper Finetuned by Zeroth-Korean Dataset](https://huggingface.co/seastar105/whisper-medium-ko-zeroth)
- [Zeroth-Korean Dataset](https://huggingface.co/datasets/kresnik/zeroth_korean/viewer/default/train?row=0)
