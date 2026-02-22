pip install -r requirements.txt

# Training script with configurable hyperparameters
python src/train_task1.py --epochs 15 --batch_size 128 --lr 1e-3 --scheduler cosine --use_augmentation
#  Evaluation script: all metrics + plots + failure cases
python src/eval_task1.py --model_path models/task1_best.pt --history_path reports/task1/train_history.json
python src/task2_report_generation.py
python src/task3_retrieval.py

