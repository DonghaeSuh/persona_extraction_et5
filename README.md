# persona_extraction_et5

## 실행 방법

### 1. requirements
cuda version : 11.4, linux-64 에서
```
# $ conda create --name [가상환경이름] --file requirements.txt
```

### 2. Config 설정
- configs폴더 내에 config 파일을 생성
  ```
  {"model_name": "gogamza/kobart-base-v2", # 모델 이름
  "model_detail" : "kobart-baeline-rouge-bleu-by-val_bleu_avg", # 모델 세부 설명
  "resume_path" : "./checkpoints/gogamza/kobart-base-v2kobart-baeline-rouge-bleu-by-val_bleu_avg.ckpt", # 만약 학습을 이어서 하고 싶을 경우 checkpoint 파일 경로

  "wandb_entity" : "gypsi12", # wandb entity이름
  "wandb_project" : "persona_extraction", # wandb 프로젝트 이름
  "wandb_run_name" : "kobart-baeline-rouge-bleu-by-val_bleu_avg", # wandb상 실행 이름
  
  "batch_size": 16, # 배치 크기
  "shuffle": true, # 학습 데이터 shuffle 여부
  "learning_rate":1e-5, # learning rate
  "epoch": 10, # 최대 epoch
  
  "train_path":"./data/train/train.csv", # 학습 데이터 경로
  "dev_path":"./data/val/validation.csv", # 검증 데이터 경로
  "test_path":"./data/val/validation.csv", # 테스트 데이터 경로
  "predict_path":"./data/val/validation_csv"} # 예측 데이터 경로
  ```

### 3. 학습
```
python train.py
```

### 4. 학습 결과
- 학습 결과는 ./best_model/ 경로에 .pt로 저장됨
- checkpoint들을 ./checkpoints/ 경로에 .ckpt로 저장됨
