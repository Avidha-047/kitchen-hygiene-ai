# Trained model

`best.pt` is the fine-tuned YOLOv8n cockroach detector (training run `kitchen416`).

| Split | Images | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| Test (never seen in training) | 67 | 0.82 | 0.81 | 0.87 | 0.36 |

Trained on Google Colab (T4 GPU) from YOLOv8n pretrained on COCO, with the first 10 layers frozen,
416 px images, up to 60 epochs with early stopping (stopped at 42, best at 27).
Data: 310 labelled photos from [nafiz5662/cockroach-detection](https://github.com/nafiz5662/cockroach-detection), split 207 / 36 / 67.
