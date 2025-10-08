## AI based korean Named Entity Recognizer

팀원 : [허명범](https://github.com/MyungBeomHer)

### 프로젝트 주제 
한국어 개체명 인식기

### 프로젝트 언어 및 환경
프로젝트 언어 : Pytorch

### Dataset
- [NER Dataset from 한국해양대학교 자연언어처리 연구실](https://github.com/kmounlp/NER)

### NER tagset
- 총 8개의 태그가 있음
    - PER: 사람이름
    - LOC: 지명
    - ORG: 기관명
    - POH: 기타
    - DAT: 날짜
    - TIM: 시간
    - DUR: 기간
    - MNY: 통화
    - PNT: 비율
    - NOH: 기타 수량표현
- 개체의 범주 
    - 개체이름: 사람이름(PER), 지명(LOC), 기관명(ORG), 기타(POH)
    - 시간표현: 날짜(DAT), 시간(TIM), 기간 (DUR)
    - 수량표현: 통화(MNY), 비율(PNT), 기타 수량표현(NOH)

## ➡️ Data Preparation
```bash
cd data_in/NER-master/
unzip 말뭉치\ -\ 형태소_개체명/.zip
```

### Requirements
```bash
pip install -r requirements.txt
```

### train
```bash
python train_bert_crf.py 
```

### inference
```bash
python inference.py 
```


### Model
<p align="center">
  <img src="/figure/model.png" width=100%> <br>
</p>

```
#KobertCRF + FRU-Adapter
class KobertCRF(nn.Module):
    """ KoBERT with CRF FRU-Adapter"""
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertCRF, self).__init__()

        if vocab is None:
            self.bert, self.vocab = get_pytorch_kobert_model()
        else:
            self.bert = BertModel(config=BertConfig.from_dict(bert_config))
            self.vocab = vocab

        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_labels=num_classes)
        self.pad_id = getattr(config, "pad_id", 1)  # 기본 1

        self.tsea_blocks = nn.ModuleList([
            FRU_Adapter(embded_dim=768) for _ in range(12)
        ])

    def forward(self, input_ids, token_type_ids=None, tags=None):
        # --- 1) BERT attention mask (2D -> extended additive mask) ---
        pad = self.vocab.token_to_idx[self.vocab.padding_token]
        mask_2d = input_ids.ne(pad).to(dtype=self.bert.embeddings.word_embeddings.weight.dtype)  # [B, L]
        extended_attention_mask = self.bert.get_extended_attention_mask(
            mask_2d, mask_2d.shape, device=input_ids.device
        )  # [B,1,1,L], tokens: 0.0, pads: -10000.0

        # --- 2) 임베딩 + (동결된) 인코더 + FRU 병렬잔차 ---
        hidden_states = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)  # [B,L,768]

        for i, encoder_layer in enumerate(self.bert.encoder.layer):
            prev = hidden_states
            out = encoder_layer(hidden_states=hidden_states, attention_mask=extended_attention_mask)
            x = out[0] if isinstance(out, (tuple, list)) else out
            x = x + self.tsea_blocks[i](prev)  # FRU Adapter
            # x = x + self.tsea_blocks[i](x)  # FRU Adapter
            hidden_states = x

        last_encoder_layer = self.dropout(hidden_states)
        emissions = self.position_wise_ff(last_encoder_layer)  # [B,L,num_classes]

        # --- 3) CRF용 mask (bool, [B,L]) ---
        crf_mask = input_ids.ne(self.pad_id)  # True=유효토큰, False=패딩
    
        max_len = input_ids.size(1)   # y_real과 동일한 목표 길이
        pad_val = self.pad_id         # 패딩값 (어차피 acc 계산에서 pad는 마스크됨)

        def _pad_paths(paths, tgt_len, pad_val):
            out = []
            for p in paths:
                if len(p) < tgt_len:
                    p = p + [pad_val] * (tgt_len - len(p))
                else:
                    p = p[:tgt_len]
                out.append(p)
            return out

        if tags is not None:
            log_likelihood = self.crf(emissions, tags, mask=crf_mask.to(torch.uint8))
            seq = self.crf.viterbi_decode(emissions, mask=crf_mask.to(torch.uint8))
            seq = _pad_paths(seq, max_len, pad_val)                       # ★ 패딩
            sequence_of_tags = torch.tensor(seq, device=input_ids.device) # 텐서로 변환
            return log_likelihood, sequence_of_tags
        else:
            seq = self.crf.viterbi_decode(emissions, mask=crf_mask.to(torch.uint8))
            seq = _pad_paths(seq, max_len, pad_val)                       # ★ 패딩
            sequence_of_tags = torch.tensor(seq, device=input_ids.device)
            return sequence_of_tags

```
[model/net.py](model/net.py)

- Benchmark (NER Dataset)

|Model|Params|MacroAvg F1 score|
|:------:|:------:|:---:|
|KoBERT|92.21M|0.8554|
|KoBERT+CRF|92.21M|0.8756||
|KoBERT+BiLSTM+CRF|95.75M|0.8659||
|KoBERT+Temporal-Adapter+CRF|95.38M|0.8623||
|**KoBERT+FRU-Adapter+CRF**|95.38M|**0.8769**||

### Reference Repo
- [SKTBrain KoBERT](https://github.com/SKTBrain/KoBERT)
- [Finetuning configuration from huggingface](https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_multiple_choice.py)
- [SKTBrain KoBERT Error revise](https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf)
- [KoBERT using NER](https://github.com/eagle705/pytorch-bert-crf-ner/tree/master?tab=readme-ov-file)
- [FRU-Adapter](https://github.com/SeoulTech-HCIRLab/FRU-Adapter)
