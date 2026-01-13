# Cooking Time ML (kitchen → packed)

Проект для обучения модели, предсказывающей `cooking_minutes = packed_time - kitchen_time` на основе:
- метаданных заказа (точка, город, способ доставки, время и т.п.)
- состава заказа из `dump_item` (хешированные sparse-признаки + агрегаты)

## 1) Требования

- Python 3.10+ (рекомендуется 3.11)
- Доступ к MariaDB (DataLens) для первичной выгрузки
- Достаточно диска: датасет может быть большим (миллионы строк)

### Установка зависимостей

Если используете `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```


## 2) Пример запуска
```bash
python prepare_dataset.py \
  --input datasets/orders_cooking_raw \
  --output datasets/prepared_no_preorders \
  --hash-items \
  --hash-dim 65536
```
```bash
python train_compare_models.py \
  --base-dir datasets/prepared_no_preorders/prepared_base \
  --items-dir datasets/prepared_no_preorders/prepared_items_hash \
  --out-dir model_out_no_preorders \
  --use-log-target \
  --seed 42 \
  --ridge-alpha 10 \
  --sgd-alpha 1e-5 \
  --sgd-loss huber \
  --sgd-epsilon 1.35
```