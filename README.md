# SmolVLA Memory Experiments

Исследовательский задел для добавления компактной истории наблюдений к SmolVLA.

## Первый этап

Пока проект не изменяет LeRobot и не требует GPU. Мы проверяем:

1. формы наблюдений и действий;
2. работу memory-модуля на синтетической последовательности;
3. воспроизводимый baseline без памяти и с compressed memory.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Smoke test

```bash
python -m experiments.00_memory_smoke
```

Detailed experiment commands and parameter explanations are in [RUNBOOK.md](RUNBOOK.md).

Позже LeRobot подключается отдельным окружением:

```bash
git clone https://github.com/huggingface/lerobot.git external/lerobot
pip install -e 'external/lerobot[smolvla]'
```
