# leo50s-montage

Плагин Claude Code для трейлера из AI-клипов: один свет и лук на все клипы, монтаж по файлу `trailer.json`, реплики остаются целыми, музыка и удары синтезируются без лицензионных треков. Вырос из трейлера «Землянка»: его точная сборка лежит в плагине как пример, протокол приёмки — [tests/acceptance/zemlyanka-v6.md](tests/acceptance/zemlyanka-v6.md).

## Четыре скилла

- `trailer-build` — «собери трейлер», «сделай тизер из этих клипов», «добавь титр», «переставь сцены», «пересобери»: сборка и правка по `trailer.json`.
- `match-grade` — «сделай единый грейд», «выровняй яркость», «этот клип темнее остальных», «сделай ч/б-нуар»: гамма и контраст каждого клипа подбирает решатель.
- `speech-safe-cuts` — «не обрезай фразу», «должно быть разборчиво», «подтверди, что реплика целая»: границы по огибающей RMS, проверка через whisper.
- `synth-score` — «сделай музыку», «добавь сердцебиение», «удар на титре», «музыка глушит реплику», «какая громкость в LUFS»: слои `score` и удары `hit`.

## Требования

- macOS на Apple Silicon.
- `ffmpeg` и `ffprobe` из homebrew (`drawtext` не нужен).
- Python 3.13+.
- `uv`: из него ставится отдельное окружение whisper на Python 3.12.
- Системные шрифты (для карточек и титров).

## Установка

```
/plugin marketplace add leogodnik/leo50s-montage
/plugin install leo50s-montage@leo50s
bash ~/.claude/plugins/cache/leo50s/leo50s-montage/0.1.1/scripts/setup_env.sh
```

Третья команда создаёт окружения в `~/.cache/leo50s-montage` (`venv` и `venv-whisper`) и заранее скачивает модель whisper `medium` (около 1,5 ГБ). Другое место задаёт переменная `LEO50S_HOME`; она должна быть выставлена и при установке, и при работе. Без whisper: `setup_env.sh --no-whisper`. Номер версии в пути меняется с обновлением плагина.

## Как начать

1. Осмотреть клипы:
   ```
   ~/.cache/leo50s-montage/venv/bin/python <плагин>/skills/trailer-build/scripts/probe_clips.py <папка с клипами>
   ```
2. Положить `<проект>/_сборка/trailer.json`. Минимальный тизер из шести сегментов:
   ```json
   {
     "clips": {"a": "../clips/a.mp4", "b": "../clips/b.mp4"},
     "fps": 24,
     "size": [1920, 1080],
     "look": "noir",
     "edl": [
       {"id": "s00", "kind": "black", "d": 1.0},
       {"id": "s01", "kind": "clip", "src": "a", "ss": 2.4, "d": 5.0, "ov": 0.6, "speech": true},
       {"id": "s02", "kind": "card", "text": "ОНИ ЖДАЛИ ПРИКАЗА", "d": 3.0, "ov": 0.5},
       {"id": "s03", "kind": "clip", "src": "b", "ss": 0.0, "d": 4.0, "ov": 0.4, "hit": "boom"},
       {"id": "s04", "kind": "title", "text": "ЗЕМЛЯНКА", "d": 4.0, "style": "painted", "ov": 0.5},
       {"id": "s05", "kind": "black", "d": 1.5, "ov": 0.5}
     ],
     "score": {
       "layers": [{"instrument": "drone", "from": "s00", "to": "s04", "gain": 0.3}],
       "hits": {"boom": 0.7}
     }
   }
   ```
   Все поля описаны в `skills/trailer-build/references/project-format.md`.
3. Собрать и проверить (из папки `_сборка`):
   ```
   PY=~/.cache/leo50s-montage/venv/bin/python
   $PY <плагин>/skills/trailer-build/scripts/build.py trailer.json all
   $PY <плагин>/skills/speech-safe-cuts/scripts/transcribe_words.py trailer.json
   $PY <плагин>/skills/speech-safe-cuts/scripts/check_cuts.py trailer.json
   $PY <плагин>/skills/speech-safe-cuts/scripts/verify_mix.py trailer.json --transcribe
   $PY <плагин>/skills/trailer-build/scripts/contact_sheet.py trailer.json out/sheet.jpg
   ```
   `build.py all` создаёт папки `seg/`, `cards/`, `wav/` и `out/` рядом с `trailer.json`;
   мастер — `out/trailer.mp4` (имя задаёт `output.name`). `transcribe_words.py` пишет
   `words.json`: он нужен проверкам, потому что в примере есть `"speech": true`. Первый
   запуск скачивает модель whisper `medium` (около 1,5 ГБ), если её уже не скачал
   `setup_env.sh`, и может идти несколько минут.

Обычно команды запускает сам Claude по фразе из списка скиллов; руками их гонять не обязательно.

## Пример

`skills/trailer-build/references/example-trailer.json` — «Землянка»: 25 сегментов, 66,42 с. Пояснения — в `example-trailer.md` рядом. Приёмка на исходных клипах: 1594 кадра, звук совпал с оригиналом до сэмпла, `check_cuts` и `verify_mix` показали `problems: 0`.

## Тесты

```
~/.cache/leo50s-montage/venv/bin/python -m pytest -q -m "not slow"
~/.cache/leo50s-montage/venv/bin/python -m pytest -q -m slow    # тест whisper, около 30 с
```

## Лицензия

MIT.
