# Рецепты партитуры

Каждый рецепт — фрагмент `trailer.json`: слить с проектом (сегменты по `id`, слои добавить
в `score.layers`), затем `build.py <trailer.json> audio`, `mux`, `audio_report.py`.
Id взяты из `trailer-build/references/example-trailer.json`, подставь свои.

## Нарастание к взрыву

Тиканье до перелома, райзер на слоу-мо, взрыв со свистом бомбы на белом.

```json
{
  "edl": [
    {"id": "s19", "kind": "clip", "src": "c4", "ss": 18.2, "d": 4.0, "trans": "cut",
     "fx": ["slow4", "push"], "hit": "riser"},
    {"id": "s20", "kind": "white", "d": 1.5, "ov": 0.6,
     "hit": {"type": "explosion", "whistle_from": "s19"}}
  ],
  "score": {"layers": [
    {"instrument": "ticks", "from": "s07", "to": "s10", "params": {"gap": 0.5},
     "gain": 0.18, "trim_end": 0.35, "attack": 0, "release": 0}
  ]}
}
```

`trim_end` гасит тиканье за 0.35 с до карточки, чтобы `braam` вошёл в тишину.

## Слэм названия

Рисованное название влетает на стыке, под ним `boom` с длинным `braam`.

```json
{
  "edl": [
    {"id": "s21", "kind": "title", "text": "НАЗВАНИЕ", "d": 5.0, "trans": "cut",
     "style": "painted", "bg": "black", "hit": "titleboom"}
  ],
  "score": {"hits": {"titleboom": 0.7}}
}
```

## Сердцебиение двух темпов

Медленное в тихом акте, быстрое — с первого стыка нарастания до перелома.

```json
{"score": {"layers": [
  {"instrument": "heartbeat", "from": "s02", "to": "s07", "params": {"bpm": 58},
   "gain": 0.3, "attack": 0, "release": 0},
  {"instrument": "heartbeat", "from": "s07", "to": "s10", "params": {"bpm": 84},
   "gain": 0.5, "attack": 0, "release": 0}
]}}
```

## Тишина перед титром

Слой кончается на моменте сегмента перед названием (`to`), `release` даёт короткий спад;
под названием новый слой, тихий, с быстрой атакой и долгим хвостом.

```json
{"score": {"layers": [
  {"instrument": "drone", "from": "s00", "to": "s20", "gain": 0.2,
   "attack": 3.0, "release": 0.4},
  {"instrument": "drone", "from": "s21", "to": "end", "gain": 0.16,
   "attack": 0.05, "release": 2.5}
]}}
```

## Дакинг под речь

Музыка уходит под реплику со скатами 0.25 с. 0.5 в тихом акте, 0.4 под остинато,
0.2–0.1 если verify_mix пишет `syllable lost` (решает `speech-safe-cuts`).

```json
{"edl": [
  {"id": "s11", "speech": true, "duck": 0.4},
  {"id": "s12", "speech": true, "duck": 0.2, "amb": 2.0}
]}
```

`amb` поднимает сам звук клипа; ставь его только короткому крику, который тонет.

## Громкость −17 LUFS

Мастер нормализуется по пику (`target_peak`), поэтому интегральная громкость зависит от
самого громкого удара. Пример даёт I = −16.7 LUFS при пике −0.7 dBFS.

`$PY $D/scripts/audio_report.py $T` → строки `I:`, `LRA:`, `Peak:` и PNG спектрограммы.

```json
{"score": {"hits": {"explosion": 0.8}, "master": {"target_peak": 0.95}}}
```

- Тише −19: опусти самый громкий удар (`explosion`, `bigboom`), остальное поднимется
  нормализацией; затем `target_peak` до 0.95.
- Громче −15: подними `explosion` до 1.0 или опусти слои на 10–20 %.
- После каждой правки `audio`, `audio_report.py`, пока I не в −18…−16.
