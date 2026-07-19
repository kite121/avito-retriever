# Avito Retriever

Решение задачи поиска релевантных статей справочного центра Avito. Основной
сценарий репозитория — обработать запросы из `test.f` и воспроизвести итоговый
`answer.csv`. Исследовательские ноутбуки не нужны для этого запуска.

Полное описание обработки HTML, признаков, проверки на `calibration.f` и анализа
ошибок находится в [`description/solution.md`](description/solution.md).

## Быстрый запуск на query

В репозитории уже находятся исходные файлы `articles.f`, `calibration.f` и
`test.f`. Для каждого `query_id` из `test.f` команда ниже ранжирует статьи и
записывает до 10 `article_id` в порядке релевантности.

Здесь `query` — строка `query_id, query_text` во входном `test.f`: production-
команда обрабатывает весь переданный набор query за один запуск, как требуется
для формирования submission.

```bash
git clone https://github.com/kite121/avito-retriever.git
cd avito-retriever

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install
make answer
```

Результат появится в корне проекта:

```text
answer.csv
```

Формат файла (пример строки):

```csv
query_id,answer
<query_id>,<article_id_1> <article_id_2> ... <article_id_10>
```

Первый запуск скачивает модели с Hugging Face и строит эмбеддинги. Желательно
использовать GPU; CPU тоже поддерживается, но полный прогон будет существенно
дольше. Повторный запуск с теми же входными файлами и конфигурацией использует
кеш и снова формирует валидированный CSV.

Эквивалентный вызов без `make`:

```bash
python tools/make_answer.py
```

Если Feather-файлы находятся в другой директории:

```bash
python tools/make_answer.py \
  --data-dir /path/to/candidate_data \
  --output answer.csv
```

Для принудительного пересчёта без готового кеша:

```bash
python tools/make_answer.py --force
```

Конфигурация результата зафиксирована в
[`configs/experiments/fixed_submission.yaml`](configs/experiments/fixed_submission.yaml).
Для воспроизведения отправленного результата не добавляйте флаг `--ocr`: он
запускает отдельный вариант решения и меняет fingerprint прогона.

## Проверка результата

`make answer` автоматически проверяет файл перед сохранением. Проверку также
можно повторить отдельно:

```bash
avito-validate-submission answer.csv --data-dir data/candidate_data
```

Валидатор проверяет:

- точные колонки `query_id,answer`;
- наличие ровно одной строки для каждого запроса из `test.f`;
- не более 10 уникальных статей в ответе;
- принадлежность всех идентификаторов корпусу `articles.f`.

## Что выполняется внутри

```text
HTML статей
  → структурные поля и нормализация
  → SentencePiece BM25F
  → multilingual-e5-small по фрагментам статей
  → lexical + dense kNN по размеченным calibration-запросам
  → weighted Reciprocal Rank Fusion
  → BAAI/bge-reranker-v2-m3
  → top-10 и валидация answer.csv
```

Параметры не подбираются во время production-запуска: веса полей, размеры
чанков, глубины кандидатов, коэффициенты RRF и модели читаются из фиксированного
YAML. Fingerprint строится из конфигурации, режима OCR и SHA-256 всех трёх
входных Feather-файлов, поэтому кеш от другого набора данных не переиспользуется.

## Структура проекта

```text
description/             Описание подхода, валидации и анализа ошибок
configs/                 Фиксированная конфигурация и конфиги экспериментов
data/candidate_data/     articles.f, calibration.f, test.f
src/avito_retriever/
  preprocessing/         HTML, нормализация, изображения и OCR
  tokenization/          SentencePiece
  retrieval/             BM25F, dense retrieval и query kNN
  fusion/                Weighted RRF
  reranking/             Cross-encoder reranking
  pipeline/              Воспроизводимый production-пайплайн
  evaluation/            MAP@10, recall и статистические проверки
  cli/                    Команды запуска и валидации
tools/make_answer.py      Основная точка входа для answer.csv
tests/                    Контрактные и модульные тесты
output/jupyter-notebook/  Исследовательские ноутбуки
```

## Исследования и эксперименты

Кратко: ноутбуки `00`–`06` в `output/jupyter-notebook/` содержат аудит данных,
подбор BM25F/dense/kNN/RRF, сравнение reranker-моделей, OCR-ablation и проверку
значимости на отложенной части `calibration.f`. Они нужны для обоснования выбора
архитектуры, но не для генерации итогового `answer.csv`. Инструкция по их запуску
находится в [`output/jupyter-notebook/README.md`](output/jupyter-notebook/README.md).

## Тесты

```bash
make test
```

Тесты проверяют HTML-парсер, ранжирование и fusion, метрики, разбиение
`calibration.f`, статистику и контракт итогового CSV.
