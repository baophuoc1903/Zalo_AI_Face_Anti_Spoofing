# Sample model for Zalo AI 2022: Liveness Detection challenge

## About this project:

- A Solution for Zalo AI Challenge 2022: [Liveness Detection](https://challenge.zalo.ai/portal/liveness-detection)
- Approach: EfficientNet-b0, 10-fold cross validation

## How to run:

Install dependencies:
```
pip install -r requirements.txt
```

Running 10-fold validation:
```
python main.py --eff --batch_size 256
```

Get predict:
```
python predict.py --eff --ensemble_dir [path_to_model_weight_folder] --predict_path [path_to_video_folder]
```

