# Предсказание пола по транзакциям (Sber 2026 DS)

## Цель

Предсказать пол клиента (`0` — женщина, `1` — мужчина) по истории банковских транзакций.

**Целевые метрики:** ROC AUC > 0.88 и Accuracy > 0.80

**Датасет:** [`mks-logic/gender_prediction`](https://huggingface.co/datasets/mks-logic/gender_prediction)
— 3 751 083 транзакций, 8 400 клиентов (70% train / 10% val / 20% test).

---

## Результаты

| Ветка | AUC | Accuracy | Train | Инфер | Фичи |
|---|---|---|---|---|---|
| `solution_2021_old` | 0.680 | 0.644 | 2.9s | 0.0003s | 42 |
| `solution_2026_04_15_linear` | 0.852 | 0.770 | **0.13s** | **0.005s** | 100 |
| `solution_2026_04_15_feature_importance` | 0.871 | 0.791 | 12.6s | 0.086s | 100 |
| `solution_2026_04_15_random_forest` | 0.870 | 0.790 | 16.9s | 0.080s | 100 |
| `solution_2026_04_15_lama` | 0.877 | 0.782 | 248s | 6.2s | 376 |
| `solution_2026_04_15_best_from_all_130` | 0.879 | 0.798 | 90.2s | — | 130 |
| `solution_2026_04_15_best_from_all_200` | 0.882 ✓ | 0.801 ✓ | 128.4s | — | 200 |
| `solution_2026_04_06` | 0.882 ✓ | 0.802 ✓ | — | — | 376 |
| `solution_2026_04_15_holidays_200` | 0.880 ✓ | 0.801 ✓ | 113.9s | — | 200 |
| `solution_2026_04_15_holidays_230` | 0.881 ✓ | 0.799 | 166.3s | — | 230 |
| `solution_2026_04_15_gender_embedding` | 0.881 ✓ | 0.803 ✓ | 111.5s | — | 200 |
| `solution_2026_04_15_mcc_interactions` | 0.883 ✓ | 0.802 ✓ | 112.2s | — | 200 |
| `solution_2026_04_15_super_100` | 0.874 | 0.789 | 87.7s | — | 100 |
| `solution_2026_04_15_super_200` | 0.882 ✓ | **0.807 ✓** | 143.2s | — | 200 |
| **`solution_2026_04_15_super_all`** | **0.885 ✓** | **0.803 ✓** | 466.0s | — | 685 |

---

## Типы признаков

| Группа | Кол-во | Описание |
|---|---|---|
| A. Base | 376 | Amount stats, temporal, per-MCC/TR, target encoding (solution.py) |
| B. Hour shares + pos/neg | 30 | Per-hour normalized shares (0-23), pos/neg min/max/median |
| C. MCC × время | 160 | top-20 MCC × morning/afternoon/evening/night/weekend/q25/q75/median |
| D. MCC depth | 80 | top-20 MCC × neg_ratio, max_amount, amount_cv, cnt_ratio |
| E. Праздники | 30 | 10 праздников × mean/std/near7_ratio (знаковое расстояние в днях) |
| F. Gender embeddings | 9 | spaCy ru_core_news_md: cosine sim описаний MCC/TR с male/female centroid |
| **Итого** | **685** | |

### Состав super top-200
`base=95, hour_shares=24, MCC×time=35, MCC_depth=12, holidays=25, gender_emb=9`

### Топ-5 признаков по importance
`mcc_code_te`, `dow0_ratio`, `dow6_ratio`, `hour_std`, `afternoon_ratio`

---

## Окружение

```bash
conda activate sber
python download_dataset.py  # скачать данные в ./data/
```

Скрипты решений — в соответствующих ветках.

---

## Идеи для продолжения

- Word2Vec / seq2vec по последовательности MCC-кодов (транзакции как «текст»)
- CatBoost, XGBoost, стекинг нескольких моделей
- Neural network с learnable embedding для MCC/tr_type
- Bayesian hyperparameter search (Optuna)
- Per-tr_type × time interactions (аналог MCC × time)
- Больше MCC для взаимодействий (сейчас top-20, можно top-40)
