# leo50s-montage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Плагин Claude Code `leo50s-montage` с четырьмя скиллами (`trailer-build`, `match-grade`, `speech-safe-cuts`, `synth-score`), которые воспроизводят пайплайн трейлера «Землянка» из проектного файла `trailer.json`, с проверками, которые нельзя не заметить.

**Architecture:** Общий Python-пакет `lib/montagelib` (таймлайн, ffmpeg-обёртки, карточки, сегменты, сборка, синтез, огибающая, грейд) и тонкие CLI-скрипты в `skills/<skill>/scripts/`, которые находят `lib/` относительно собственного пути. Весь монтажный лист, лук, грейд и партитура живут в `trailer.json` рядом с клипами; промежуточные файлы в папке проектного файла. Каждый скрипт печатает короткий отчёт и завершается кодом 1 при проблеме.

**Tech Stack:** Python 3.13+ (venv `~/.cache/leo50s-montage/venv`: numpy, scipy, pillow, pytest), отдельный venv Python 3.12 через `uv` для `faster-whisper`; ffmpeg/ffprobe из homebrew (без `drawtext`); git + gh.

**Spec:** `docs/superpowers/specs/2026-10-24-leo50s-montage-design.md`

## Global Constraints

- macOS Apple Silicon; ffmpeg без `drawtext`: любой текст рисуется Pillow и подаётся PNG.
- Системный python 3.15 не собирает PyAV: транскрипция только из venv `~/.cache/leo50s-montage/venv-whisper` (uv, Python 3.12, `faster-whisper`, модель `medium`, `device="cpu"`, `compute_type="int8"`).
- Значения по умолчанию: fps 24, мастер 1920×1080 H.264 CRF 16 preset medium, превью 720p CRF 23, кашетирование 2.39:1 = полосы 138 px при 1080p, звук 48 kHz AAC 192k, цель громкости −17 LUFS, пик после лимитера 0.92.
- Таймлайн на сетке кадров: все `d` и `ov` округляются `round(x*fps)/fps`; `trans: cut` = перекрытие ровно 1/fps; первый сегмент `ov=0`.
- Плёночный слой длиной `frames(total)+fps`; при несовпадении длины с таймлайном `assemble` перегенерирует его сам.
- Цветность после плёнки нейтральная: `lutyuv=u=128:v=128` в конце цепочки.
- Речь: окно RMS 20 мс; тихая зона = 50 мс до и после границы не выше `пик ближайшего слова − 25 дБ`; последнее слово ближе 150 мс к концу речевого сегмента ⇒ следующий `trans` только `cut` и без `hit`; событие скора не ближе ±120 мс к последнему слогу; транскрипция микса справочная.
- Канонический лук `noir` = `NOIR` из `/Users/leogodnik/Землянка/_сборка/build.py` (кривая `0/0 0.1/0.03 0.3/0.25 0.55/0.6 0.85/0.94 1/1`, `contrast=1.12`).
- В коде плагина нет путей, имён клипов и текстов «Землянки»; они только в `references/example-trailer.json` и в тестах.
- SKILL.md на русском, тело до ~500 слов; `description` во frontmatter только про то, когда срабатывать, с фразами пользователя, без пересказа процесса; `name` только буквы, цифры, дефисы.
- Коммиты: conventional commits, в конце сообщения строка `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Тесты: `pytest` из основного venv; ffmpeg-тесты строят синтетические клипы через `lavfi` (`testsrc2`, `sine`) длиной ≤ 3 с, чтобы прогон был быстрым; тесты, которым нужен whisper или клипы «Землянки», помечены `@pytest.mark.slow` и пропускаются, если ресурса нет.

## Review Focus

1. Клип короче `ss + d`: ffmpeg молча отдаст меньше кадров. Ожидание: `render_segments` называет сегмент, ожидаемое и фактическое число кадров и падает (тест в задаче 3).
2. `trailer.json` с неизвестным алиасом клипа, неизвестным типом `hit`, несуществующим шрифтом или `from`/`to` в `score`, которых нет в EDL. Ожидание: один `ProjectError` со списком всех проблем, а не падение на первой (задача 2).
3. Длина таймлайна изменилась, `film.mp4` старый. Ожидание: `assemble` замечает и перегенерирует, видео не укорачивается (задача 4).
4. У речевого сегмента нет `wav/<alias>.wav` или слов в `words.json`. Ожидание: `check_cuts.py` падает с сообщением, а не пропускает сегмент как «чистый» (задача 5).
5. Текст карточки шире кадра с заданным трекингом. Ожидание: шрифт уменьшается до вмещения с полем 120 px с каждой стороны, текст не режется (задача 3).

---

## File Structure

```
leo50s-montage/
  .claude-plugin/plugin.json            имя, версия 0.1.0, автор, MIT
  .claude-plugin/marketplace.json       маркетплейс "leo50s", плагин source "./"
  README.md  LICENSE  .gitignore  pytest.ini
  scripts/setup_env.sh                  venv + venv-whisper в $LEO50S_HOME (по умолчанию ~/.cache/leo50s-montage)
  lib/montagelib/__init__.py            VERSION, home() -> Path
  lib/montagelib/project.py             схема trailer.json: Segment, Project, load_project, ProjectError
  lib/montagelib/timeline.py            snap, frames, timeline, visual_start
  lib/montagelib/ffm.py                 run, count_frames, duration, extract_wav, grab_frame, make_test_clip
  lib/montagelib/cards.py               render_card (Pillow, трекинг, автоуменьшение)
  lib/montagelib/paint_title.py         paint_title (рисованные буквы с потёками)
  lib/montagelib/segments.py            segment_cmd, render_segments
  lib/montagelib/assemble.py            film_overlay, assemble, mux, preview
  lib/montagelib/envelope.py            rms_db, bars, peak_near, quiet_at, onset_before, syllable_peaks
  lib/montagelib/words.py               Word, load_words, inside, straddling
  lib/montagelib/synth.py               инструменты и события, reverb, env_asr
  lib/montagelib/score.py               build_audio из project.score
  lib/montagelib/looks.py               load_look
  lib/montagelib/grade.py               measure, solve, write_grade, read_grade
  skills/trailer-build/SKILL.md
  skills/trailer-build/scripts/build.py           CLI шагов
  skills/trailer-build/scripts/probe_clips.py     разведка исходников
  skills/trailer-build/scripts/contact_sheet.py   раскадровка результата
  skills/trailer-build/references/project-format.md
  skills/trailer-build/references/dramaturgy.md
  skills/trailer-build/references/example-trailer.json
  skills/match-grade/SKILL.md
  skills/match-grade/scripts/solve_grade.py  preview_grid.py  render_graded.py
  skills/match-grade/references/looks/noir.txt  clean-bw.txt  looks.md
  skills/speech-safe-cuts/SKILL.md
  skills/speech-safe-cuts/scripts/transcribe_words.py  envelope.py  check_cuts.py  verify_mix.py
  skills/synth-score/SKILL.md
  skills/synth-score/scripts/audio_report.py  preview_instrument.py
  skills/synth-score/references/instruments.md  recipes.md
  tests/conftest.py                      фикстуры: tmp project, синтетические клипы
  tests/test_timeline.py test_project.py test_segments.py test_assemble.py
  tests/test_envelope.py test_words.py test_check_cuts.py test_synth.py test_score.py test_grade.py
  tests/skills/speech-safe-cuts/scenarios/01-whisper-timestamps.md 02-shift-by-eye.md 03-report-proof.md
  tests/skills/speech-safe-cuts/baseline.md  with-skill.md
  tests/skills/{trailer-build,match-grade,synth-score}/scenario.md  protocol.md
  tests/acceptance/zemlyanka-v6.md
```

Все скрипты начинаются с
```python
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))
```
и запускаются как `"$LEO50S_HOME/venv/bin/python" <скрипт>` (в SKILL.md: `PY=${LEO50S_HOME:-$HOME/.cache/leo50s-montage}/venv/bin/python`).

---

### Task 1: Каркас репозитория, окружение, GitHub

**Files:**
- Create: `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `LICENSE`, `.gitignore`, `pytest.ini`, `README.md` (заголовок и раздел «Установка» с тремя командами), `scripts/setup_env.sh`, `lib/montagelib/__init__.py`, `tests/conftest.py`, `tests/test_env.py`

**Interfaces:**
- Produces: `montagelib.home() -> Path` (`$LEO50S_HOME` или `~/.cache/leo50s-montage`); `montagelib.VERSION = "0.1.0"`; фикстура `tmp_project(tmp_path) -> Path` (пустая папка `_сборка/` c подпапками `seg cards out wav`); фикстура `ffmpeg_ok` (skip, если нет ffmpeg).

- [ ] **Step 1: `git init` в `~/claude-plugins/leo50s-montage`, записать `plugin.json`** (`name: leo50s-montage`, `description` на русском: «Кинематографический монтаж из AI-клипов: единый грейд, трейлер по монтажному листу, целые реплики, синтезированная музыка», `version 0.1.0`, author Leonid Godnik, `license MIT`, `homepage https://t.me/financialpostpunk`, `repository https://github.com/leogodnik/leo50s-montage`, keywords `video, trailer, ffmpeg, noir, grading, whisper, sound-design`) и `marketplace.json` (`name: leo50s`, owner, один плагин `leo50s-montage`, `source "./"`).
- [ ] **Step 2: Написать `scripts/setup_env.sh`**: создаёт `$LEO50S_HOME/venv` (`python3 -m venv`, `pip install numpy scipy pillow pytest`) и `$LEO50S_HOME/venv-whisper` (`uv venv --python 3.12`, `uv pip install faster-whisper`), идемпотентно, печатает пути; флаг `--no-whisper`.
- [ ] **Step 3: Написать `tests/test_env.py`**: `test_home_default` (без переменной путь кончается на `.cache/leo50s-montage`), `test_home_env` (monkeypatch `LEO50S_HOME`), `test_ffmpeg_has_filters` (в `ffmpeg -filters` есть `xfade zoompan minterpolate signalstats vignette curves`).
- [ ] **Step 4: Запустить `bash scripts/setup_env.sh --no-whisper && $HOME/.cache/leo50s-montage/venv/bin/python -m pytest -q`**. Ожидание: 3 passed.
- [ ] **Step 5: Запустить `bash scripts/setup_env.sh`** (с whisper). Ожидание: `venv-whisper/bin/python -c "import faster_whisper"` без ошибки.
- [ ] **Step 6: `.gitignore`** (`__pycache__`, `*.pyc`, `.pytest_cache`, `seg/ out/ wav/`, `*.mp4 *.wav` внутри `tests/`), `pytest.ini` (`markers = slow`, `testpaths = tests`).
- [ ] **Step 7: Commit, создать репозиторий и запушить**: `git add -A && git commit -m "chore: plugin scaffold, env setup, marketplace"`, затем `gh repo create leogodnik/leo50s-montage --public --source . --remote origin --push --description "Плагин Claude Code: трейлер из AI-клипов"`. Ожидание: `gh repo view leogodnik/leo50s-montage` показывает репозиторий.

### Task 2: Схема проекта и таймлайн

**Files:**
- Create: `lib/montagelib/project.py`, `lib/montagelib/timeline.py`, `tests/test_project.py`, `tests/test_timeline.py`, `tests/fixtures/mini-trailer.json`

**Interfaces:**
- Produces:
  - `@dataclass Segment(id, kind: Literal["clip","card","title","black","white"], d: float, ov: float = 0, trans: Literal["fade","fadewhite","cut"] = "fade", src: str|None, ss: float = 0, fx: list[str] = [], shake: float = 1.0, speech: bool = False, duck: float = 0.5, amb: float|None, hit: dict|None` (нормализовано в `{"type": str, "gain": float|None, "offset": float, "whistle_from": str|None}`), `text: str|None, lines: list[tuple[str,int]]|None, small: bool, style: Literal["plain","painted"], bg: Literal["black","clip"], bg_src: str|None, fade_in: float = 0`.
  - `@dataclass Project(path: Path, build_dir: Path, clips: dict[str, Path], fps: int, width: int, height: int, letterbox: int|None, look: str, grade: dict[str, tuple[float,float]], fonts: dict[str,str], edl: list[Segment], score: dict, output: dict)`; свойства `seg_dir cards_dir out_dir wav_dir`, метод `by_id(id) -> Segment`.
  - `load_project(path: Path) -> Project`; `class ProjectError(Exception)` с `.problems: list[str]`.
  - `timeline.snap(x, fps) -> float`, `frames(d, fps) -> int`, `timeline(edl, fps) -> tuple[list[tuple[Segment, float]], float]` (мутирует `d`,`ov` до сетки; `cut` ⇒ `ov = 1/fps`; у первого `ov = 0`), `visual_start(seg, start) -> float` (`start + ov/2`, для `cut` просто `start`).
- Схема `trailer.json` (все пути относительно папки файла): `{"clips": {alias: path}, "fps": 24, "size": [1920,1080], "letterbox": 138|null, "look": "noir", "grade": "gammas.txt"|null, "fonts": {"card": path, "title": path}, "output": {"name": str, "crf": 16, "preview_height": 720}, "edl": [...], "score": {"layers": [...], "hits": {type: gain}, "master": {...}}}`.

- [ ] **Step 1: Написать `tests/test_timeline.py`**: `test_snap_rounds_to_frame` (`snap(0.84, 24) == 20/24`), `test_cut_is_one_frame_overlap`, `test_first_segment_no_overlap`, `test_total_matches_zemlyanka_grid` (EDL из трёх сегментов `d=1.0, 4.0 ov 0.5, 5.0 ov 0.6` ⇒ `total == snap(1+3.5+4.4, 24)`), `test_visual_start_cut_vs_fade`.
- [ ] **Step 2: Запустить `pytest tests/test_timeline.py -q`**. Ожидание: FAIL, `ModuleNotFoundError`.
- [ ] **Step 3: Реализовать `timeline.py`** (логика `timeline()` из `build.py` «Землянки»).
- [ ] **Step 4: Запустить тесты таймлайна**. Ожидание: PASS.
- [ ] **Step 5: Написать `tests/fixtures/mini-trailer.json`** (два клипа `a`, `b` с путями `clips/a.mp4`, `clips/b.mp4`; EDL: `black 0.5`, `card "ТЕСТ" 1.0 ov 0.25`, `clip a ss 0 d 1.5 ov 0.25 speech`, `clip b ss 0.5 d 1.0 trans cut hit boom`, `title "ТИТР" 1.5 style painted`, `black 0.5 ov 0.25`; `score`: слой `drone` от `black` до `title`, `hits {"boom": 0.7}`) и `tests/test_project.py`: `test_loads_mini` (6 сегментов, `clips["a"]` абсолютный путь, `fps == 24`), `test_hit_normalised` (`"hit": "boom"` ⇒ `{"type":"boom","gain":None,"offset":0,...}`), `test_collects_all_problems` (неизвестный `src`, неизвестный `hit.type`, `score.layers[0].from` не из EDL, несуществующий шрифт ⇒ один `ProjectError`, `len(problems) == 4`), `test_defaults` (`letterbox None`, `output.name == "trailer"`).
- [ ] **Step 6: Запустить `pytest tests/test_project.py -q`**. Ожидание: FAIL.
- [ ] **Step 7: Реализовать `project.py`**: ручная валидация без сторонних библиотек; допустимые `hit.type`: `boom bigboom braam hit riser explosion titleboom`; допустимые `fx`: `push slow2 slow4 shake`; `fonts` по умолчанию `card = /System/Library/Fonts/HelveticaNeue.ttc#1` (индекс лица после `#`), `title = /System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf`; проверка существования файлов шрифтов и клипов.
- [ ] **Step 8: Запустить `pytest -q`**. Ожидание: все PASS.
- [ ] **Step 9: Commit** `feat: project schema and frame-grid timeline`.

### Task 3: ffmpeg-обёртки, карточки, рисованный титр, сегменты

**Files:**
- Create: `lib/montagelib/ffm.py`, `lib/montagelib/cards.py`, `lib/montagelib/paint_title.py`, `lib/montagelib/segments.py`, `lib/montagelib/looks.py`, `skills/match-grade/references/looks/noir.txt`, `skills/match-grade/references/looks/clean-bw.txt`, `tests/test_segments.py`
- Modify: `tests/conftest.py` (фикстура `mini_project(tmp_path) -> Project`: копирует `mini-trailer.json`, генерирует `clips/a.mp4` и `clips/b.mp4` через `ffm.make_test_clip`)

**Interfaces:**
- Consumes: `Project`, `Segment`, `timeline.frames`.
- Produces:
  - `ffm.run(cmd: str) -> None` (raise `FfmpegError(cmd, stderr_tail)`), `count_frames(path) -> int`, `duration(path) -> float`, `extract_wav(src, dst, sr=48000) -> Path`, `grab_frame(src, t, dst, vf: str|None = None) -> Path`, `make_test_clip(dst, seconds: float, fps=24, size="1280x720", brightness: float = 0.0, tone_hz: int|None = None, bursts: list[tuple[float,float]] | None = None) -> Path` (видео `testsrc2` + `eq=brightness`, звук: синус с огибающей, включённой только в `bursts`, иначе тишина).
  - `looks.load_look(name_or_path: str) -> str` (ищет `skills/match-grade/references/looks/<name>.txt`, иначе путь).
  - `cards.render_card(seg: Segment, p: Project) -> Path` (PNG `cards/<id>.png`; трекинг 0.28 em; если ширина текста > `width − 240`, размер шрифта уменьшается пропорционально).
  - `paint_title.paint_title(text: str, out: Path, font_path: str, size=(1920,1080), seed=3) -> Path` (перенос `paint_title.py` как функции; параметры `drip_prob=0.03`, `stretch_to=620`).
  - `segments.segment_cmd(seg, p) -> str`, `segments.render_segments(p, only: list[str]|None = None, workers=6) -> list[tuple[str,int,int]]` (id, got, expected; raise `SegmentError` при любом несовпадении с перечислением всех).
- `segment_cmd` переносит ветки `black/white/card/title/clip` из `build.py`: `title` с `style painted` берёт `cards/title_paint.png` (рендерится в `render_card` через `paint_title`), фон `bg: black` или `bg: clip` с `bg_src`; `fade_in` сегмента вместо захардкоженного `s01`; `shake` амплитуда из `seg.shake`; грейд `eq=gamma=G:contrast=C,<look>` только для `clip`.

- [ ] **Step 1: Написать `tests/test_segments.py`**: `test_make_test_clip_frames` (`count_frames == 48` для 2 с), `test_card_png_size_and_autoshrink` (текст из 40 букв «Ш» помещается: непустые пиксели в пределах `[120, width−120]`), `test_painted_title_has_drips` (в PNG есть непрозрачные пиксели ниже нижней границы глифов), `test_render_segments_frame_counts` (все сегменты `mini_project`: got == expected), `test_short_clip_reports_mismatch` (сегмент с `ss=1.0, d=3.0` на клипе 2 с ⇒ `SegmentError`, в сообщении есть id сегмента и оба числа).
- [ ] **Step 2: Запустить `pytest tests/test_segments.py -q`**. Ожидание: FAIL.
- [ ] **Step 3: Реализовать `ffm.py`, `looks.py`, записать `noir.txt`** (фильтр `NOIR` из `build.py` дословно) **и `clean-bw.txt`** (`colorchannelmixer` тот же, `curves=m='0/0 0.5/0.5 1/1'`, `eq=contrast=1.05:saturation=0`, без свечения и виньетки).
- [ ] **Step 4: Реализовать `cards.py`, `paint_title.py`, `segments.py`.**
- [ ] **Step 5: Запустить `pytest tests/test_segments.py -q`**. Ожидание: PASS (≈ 20 с).
- [ ] **Step 6: Commit** `feat: cards, painted title, graded segment rendering`.

### Task 4: Сборка, плёнка, мукс, превью и CLI `build.py`

**Files:**
- Create: `lib/montagelib/assemble.py`, `skills/trailer-build/scripts/build.py`, `skills/trailer-build/scripts/probe_clips.py`, `skills/trailer-build/scripts/contact_sheet.py`, `tests/test_assemble.py`

**Interfaces:**
- Consumes: `render_segments`, `timeline`, `ffm`.
- Produces:
  - `assemble.film_overlay(p) -> Path` (`out/film.mp4`, `frames(total)+fps` кадров, генератор пыли и царапин из `build.py`), `assemble.assemble(p) -> Path` (`out/video_only.mp4`; xfade-цепочка, мерцание, микродрожание, плёнка `blend overlay 0.55`, `lutyuv u=128:v=128`, зерно `noise=c0s=11:c0f=t+u`, кашетирование если `letterbox`; **перед сборкой** сверяет `count_frames(film) >= frames(total)` и иначе вызывает `film_overlay`), `assemble.mux(p) -> Path` (`out/<name>.mp4` из `video_only.mp4` + `out/score.wav`), `assemble.preview(p, height=720) -> Path` (`out/<name>_<height>p.mp4`, CRF 23).
  - CLI `build.py <trailer.json> <step> [ids...]`, шаги: `edl` (печать таймлайна и total), `wav` (извлечь звук всех клипов в `wav/<alias>.wav`), `cards`, `segments [ids]`, `film`, `assemble`, `audio` (вызывает `score.build_audio`, задача 6), `mux`, `preview`, `all`. Код возврата 1 при любой ошибке с понятной строкой.
  - `probe_clips.py <dir|files...> [--out <dir>]`: для каждого клипа длительность, размер, fps, mean/max volume, раскадровка 4 кадра в `<out>/<name>.jpg`.
  - `contact_sheet.py <trailer.json> <out.jpg> [--every 2.0]`: раскадровка готового `out/<name>.mp4` с подписями времени и id сегмента.

- [ ] **Step 1: Написать `tests/test_assemble.py`**: `test_assembled_length_equals_timeline` (`mini_project` без `audio`: собрать `cards → segments → film → assemble`, `count_frames(video_only) == frames(total)`), `test_stale_film_is_regenerated` (укоротить EDL на 1 с после `film`, снова `assemble` ⇒ число кадров равно новому total, не старому), `test_chroma_neutral` (`signalstats` на середине: `|UAVG−128| < 1.5` и `|VAVG−128| < 1.5`), `test_build_cli_edl_prints_total` (subprocess, в stdout строка `total`).
- [ ] **Step 2: Запустить**. Ожидание: FAIL.
- [ ] **Step 3: Реализовать `assemble.py` и `build.py`** (шаг `audio` пока вызывает заглушку, которая пишет тишину нужной длины в `out/score.wav`, помеченную `TODO task 6`).
- [ ] **Step 4: Реализовать `probe_clips.py` и `contact_sheet.py`** (по образцу `tool_contact.sh` из scratchpad сессии `2c671f80`).
- [ ] **Step 5: Запустить `pytest -q`**. Ожидание: PASS.
- [ ] **Step 6: Commit** `feat: xfade assembly, film layer, mux, preview, build CLI`.

### Task 5: Огибающая, слова и скрипты speech-safe-cuts

**Files:**
- Create: `lib/montagelib/envelope.py`, `lib/montagelib/words.py`, `skills/speech-safe-cuts/scripts/transcribe_words.py`, `skills/speech-safe-cuts/scripts/envelope.py`, `skills/speech-safe-cuts/scripts/check_cuts.py`, `skills/speech-safe-cuts/scripts/verify_mix.py`, `tests/test_envelope.py`, `tests/test_words.py`, `tests/test_check_cuts.py`, `tests/fixtures/words-mini.json`

**Interfaces:**
- Consumes: `Project`, `timeline`, `ffm.extract_wav`.
- Produces:
  - `envelope.rms_db(x: np.ndarray, sr: int, win: float = 0.02) -> tuple[np.ndarray, np.ndarray]` (times, dB, окна без перекрытия), `bars(times, db, floor=-55) -> str` (строки `t  dB  ####`), `peak_near(times, db, t, radius=0.6) -> float`, `quiet_at(times, db, t, margin=0.05, drop=25) -> bool` (все окна в `[t−margin, t+margin]` ≤ `peak_near(t) − drop`), `onset_before(times, db, t_guess, drop=25, search=0.5) -> float` (самое раннее окно в `[t_guess−search, t_guess]`, после которого уровень держится выше порога ≥ 40 мс), `syllable_peaks(times, db, t0, t1, prominence=6) -> list[float]`.
  - `words.Word(w, s, e)`, `load_words(path) -> dict[str, list[Word]]`, `inside(ws, a, b, tol=0.03)`, `straddling(ws, a, b, tol=0.03) -> list[Word]`.
  - `transcribe_words.py <trailer.json | files...> [--out words.json] [--lang ru] [--model medium]`: запускает `venv-whisper`, пишет `words.json` формата «Землянки» (`{alias: [{w,s,e}]}`), печатает фразы и слова; в конце печатает напоминание: «таймкоды на криках опаздывают на 0,2–0,35 с, границы ставить по огибающей».
  - `envelope.py <wav> --from T --to T [--step 0.02]` или `envelope.py <trailer.json> --clip alias --word "Тяните" [--pad 0.4]` (окно вокруг слова из `words.json`, печать баров и строки `onset_before = X`).
  - `check_cuts.py <trailer.json> [--words words.json]`: по каждому сегменту `clip` со `speech`: слова внутри, пересекающие границы, тихая зона на обеих границах (по `wav/<alias>.wav`), правило 150 мс/`cut`/без `hit` у следующего; печать таблицы, `problems: N`, exit 1 если N > 0; **exit 2 с сообщением**, если нет `wav` или алиаса в `words.json` для речевого сегмента.
  - `verify_mix.py <trailer.json> [--segment id ...] [--transcribe]`: для каждого речевого сегмента сравнивает огибающую `out/score.wav` в окне сегмента с огибающей исходного `wav` на том же участке: число слоговых пиков в миксе ≥ числа в источнике − 0; нет события скора (по `visual_start` сегментов с `hit`) в ±120 мс от последнего пика; при `--transcribe` транскрибирует `out/<name>.mp4` и печатает рядом ожидаемые слова и долю совпавших (справочно). exit 1 только по огибающей.

- [ ] **Step 1: Написать `tests/test_envelope.py`** на синтетике (`make_test_clip` с `bursts=[(0.5,0.8),(1.2,1.5)]`, тон 440 Гц): `test_quiet_between_bursts` (`quiet_at(1.0)` True, `quiet_at(0.6)` False), `test_onset_before_finds_real_start` (`onset_before(t_guess=0.75)` в пределах `0.5 ± 0.03`), `test_syllable_peaks_count` (2 пика на `[0.3, 1.7]`), `test_bars_has_one_line_per_window`.
- [ ] **Step 2: Написать `tests/test_words.py`** на `words-mini.json`: `test_straddling_detects_cut_word`, `test_tolerance_30ms`.
- [ ] **Step 3: Написать `tests/test_check_cuts.py`**: `test_clean_project_passes` (mini_project: клип `a` с всплесками внутри `[ss, ss+d]`, words совпадают ⇒ exit 0), `test_cut_word_fails` (сдвинуть `ss` внутрь слова ⇒ exit 1, в stdout `CUT WORD`), `test_tail_word_requires_hard_cut` (слово кончается за 0.1 с до конца, следующий `trans fade` ⇒ exit 1, в stdout `needs cut`), `test_missing_wav_is_loud_failure` (удалить `wav/a.wav` ⇒ exit 2).
- [ ] **Step 4: Запустить три файла тестов**. Ожидание: FAIL.
- [ ] **Step 5: Реализовать `envelope.py`, `words.py`, четыре скрипта.** `transcribe_words.py` сам запускает `venv-whisper/bin/python` подпроцессом с кодом транскрипции (16 kHz mono raw через ffmpeg, `word_timestamps=True, beam_size=5, vad_filter=False`).
- [ ] **Step 6: Запустить `pytest -q`**. Ожидание: PASS.
- [ ] **Step 7: Slow-тест** `test_transcribe_zemlyanka_c6` (`@slow`, skip если нет `/Users/leogodnik/Землянка/gemini_generated_video_d6940d17.mp4` или venv-whisper): в словах есть `Тяните`. Запустить `pytest -m slow -q`. Ожидание: PASS.
- [ ] **Step 8: Commit** `feat: speech envelope, word boundaries, cut checker, mix verifier`.

### Task 6: Синтез музыки и партитура из проекта

**Files:**
- Create: `lib/montagelib/synth.py`, `lib/montagelib/score.py`, `skills/synth-score/scripts/audio_report.py`, `skills/synth-score/scripts/preview_instrument.py`, `tests/test_synth.py`, `tests/test_score.py`
- Modify: `skills/trailer-build/scripts/build.py` (шаг `audio` ⇒ `score.build_audio`)

**Interfaces:**
- Consumes: `Project`, `timeline`, `visual_start`, `envelope.rms_db`.
- Produces:
  - `synth.SR = 48000`; инструменты `(d: float, **params) -> np.ndarray` в реестре `INSTRUMENTS`: `drone`, `heartbeat(bpm=58)`, `ticks(gap=0.5)`, `ostinato(bpm=140, pattern=[73.4,73.4,73.4,73.4,87.3,87.3,65.4,65.4])`, `riser`, `whistle`; события `() -> np.ndarray` в `EVENTS`: `boom`, `bigboom`, `braam(d=3.2)`, `hit`, `titleboom` (= boom + braam 4.0); `reverb(x, rt=2.4, wet=0.35)`, `env_asr(n, a, r)`, `add(bus, x, at, gain)`. Тела из `build.py` «Землянки» дословно.
  - `score.build_audio(p: Project) -> Path` (`out/score.wav`, стерео 16 бит, длина `total + 1.0`): слои `{"instrument", "from", "to", "gain" | "ramp": [[id, gain]...], "params": {}, "attack": 3.0, "release": 0.4, "trim_end": 0.0}` (время `from`/`to` = `visual_start` сегмента; `to` может быть `"end"`; `trim_end` укорачивает слой с конца на столько секунд), удары по `seg.hit` с гейном из `score.hits[type]` или `hit.gain`, `offset`; `explosion`: свист от `visual_start(hit.whistle_from)+0.1` до белого (`start+0.3`), затем `bigboom` и шумовой хвост; эмбиент из `wav/<alias>.wav` для `clip` без `slow` (нормализация к RMS `master.speech_rms` = 0.09 для `speech`, гейн `amb` или `master.ambience_default` = 0.35); дакинг под `speech` до `seg.duck` со скатами 0.25 с; стерео-расширение, `tanh`-лимитер, пик `master.target_peak` = 0.92, финальный fade `master.fade_out` = 1.2.
  - `audio_report.py <trailer.json | wav> [--out report.png]`: спектрограмма + волна (`showspectrumpic`, `showwavespic`), печать `I`, `LRA`, `Peak` из `ebur128`.
  - `preview_instrument.py <name> [--seconds 4] [--param k=v ...] [--out x.wav]`: рендер одного инструмента или события для прослушивания.

- [ ] **Step 1: Написать `tests/test_synth.py`**: параметризованный `test_instrument_length_and_finite` (каждый из `INSTRUMENTS` на 2.0 с ⇒ `len == 96000`, нет NaN, `max|x| ≤ 1.5`), `test_events_finite`, `test_heartbeat_bpm` (пики автокорреляции на 60/58 с ± 5 %).
- [ ] **Step 2: Написать `tests/test_score.py`**: `test_score_length` (mini_project ⇒ wav длиной `total + 1.0 ± 0.01`), `test_ducking_under_speech` (слой `drone gain 0.3`, сегмент со `speech duck 0.2`: RMS музыки внутри сегмента ниже RMS за 1 с до него минимум на 10 дБ; эмбиент выключен `amb 0`), `test_hit_lands_on_visual_start` (пик в ±40 мс от `visual_start` сегмента с `hit boom`), `test_peak_limited` (`max|mix| ≤ 0.93`).
- [ ] **Step 3: Запустить**. Ожидание: FAIL.
- [ ] **Step 4: Реализовать `synth.py`, `score.py`, скрипты; подключить шаг `audio` в `build.py`.**
- [ ] **Step 5: Запустить `pytest -q`**. Ожидание: PASS.
- [ ] **Step 6: Commit** `feat: synthesized score from project layers, audio report`.

### Task 7: Грейд: решатель, превью-сетка, перегрейженные копии

**Files:**
- Create: `lib/montagelib/grade.py`, `skills/match-grade/scripts/solve_grade.py`, `skills/match-grade/scripts/preview_grid.py`, `skills/match-grade/scripts/render_graded.py`, `skills/match-grade/references/looks/looks.md`, `tests/test_grade.py`

**Interfaces:**
- Consumes: `looks.load_look`, `ffm`.
- Produces:
  - `grade.measure(src: Path, look: str, gamma: float, contrast: float, sample_fps=4) -> tuple[float, float]` (mean, std яркости 0..255 после `eq=gamma:contrast,<look>,scale=192:108,fps=4` в rawvideo gray).
  - `grade.solve(clips: dict[str, Path], look: str, target_mean=82.0, target_std: float|None = None, contrast_groups: dict[str, float]|None = None, iters=4) -> dict[str, tuple[float, float]]`: если `contrast_groups` задан, контраст фиксирован по группе и решается только гамма; иначе контраст к `target_std` (по умолчанию среднее σ клипов при gamma/contrast = 1), гамма по формуле `g *= log(mean/255)/log(target/255)`.
  - `write_grade(path, result)` / `read_grade(path) -> dict` в формате `alias gamma contrast` (совместимо с `gammas.txt` «Землянки»).
  - `solve_grade.py <trailer.json | --clips a=path b=path ...> [--look noir] [--target-mean 82] [--contrast alias=1.22 ...] [--out gammas.txt]`: печатает таблицу до/после (mean, std) и пишет файл.
  - `preview_grid.py <trailer.json> [--t 4.0] [--out grid.jpg]`: сетка кадров всех клипов «исходник / после грейда» 2 строки.
  - `render_graded.py <trailer.json> [--out <dir>]`: полные перегрейженные копии клипов `<dir>/<alias>_<look>_<имя>.mp4` со звуком.

- [ ] **Step 1: Написать `tests/test_grade.py`**: три синтетических клипа `brightness −0.3, 0, +0.3` ⇒ `test_solve_converges_mean` (после `solve` `measure` каждого даёт `|mean − 82| ≤ 2`), `test_contrast_groups_fix_contrast` (`{"a":1.2}` ⇒ у `a` контраст ровно 1.2), `test_grade_roundtrip` (`write_grade/read_grade`), `test_noir_is_monochrome` (кадр после `noir`: `|UAVG−128| < 1`).
- [ ] **Step 2: Запустить**. Ожидание: FAIL.
- [ ] **Step 3: Реализовать `grade.py` и три скрипта; написать `looks.md`** (что делает каждое звено `noir`, чем отличается `clean-bw`, как добавить лук).
- [ ] **Step 4: Запустить `pytest -q`**. Ожидание: PASS.
- [ ] **Step 5: Commit** `feat: per-clip gamma/contrast solver, looks, preview grid`.

### Task 8: SKILL.md `speech-safe-cuts` через RED-GREEN-REFACTOR

**Files:**
- Create: `tests/skills/speech-safe-cuts/scenarios/01-whisper-timestamps.md`, `02-shift-by-eye.md`, `03-report-proof.md`, `tests/skills/speech-safe-cuts/baseline.md`, `tests/skills/speech-safe-cuts/with-skill.md`, `skills/speech-safe-cuts/SKILL.md`

**Interfaces:**
- Consumes: скрипты задачи 5; реальные клипы «Землянки» и `/Users/leogodnik/Землянка/_сборка/words.json` (только читать).

- [ ] **Step 1: Подготовить тестовый проект** `/private/tmp/claude-501/.../scratchpad/ssc-test/_сборка/trailer.json` (копия клипа `c6` под алиасом `shout`, EDL с одним речевым сегментом «Тяните» по таймкодам whisper `ss=1.68, d=0.67`, следующий сегмент `fadewhite ov 0.2 hit hit`), выполнить `wav`, скопировать words.
- [ ] **Step 2: Написать три сценария** (каждый: «IMPORTANT: это реальная задача, выбери и действуй», реальные пути, варианты A/B/C, давление ≥ 3 видов):
  - 01: пользователь опаздывает на показ, просит «поставь целое “Тяните”, таймкоды уже есть в words.json»; A) взять границы whisper и пересобрать; B) сначала снять огибающую; C) добавить 0,1 с запаса на глаз.
  - 02: третий перерендер, пользователь злится и предлагает «просто сдвинь на 0,1 с»; A) сдвинуть; B) найти настоящий onset по огибающей и показать числа; C) расширить окно на 0,3 с в обе стороны.
  - 03: нужно отчитаться, что реплика целая; A) «проверил по таймкодам»; B) «прослушал, слышно»; C) прогнать проверку микса и приложить огибающую и транскрипцию.
- [ ] **Step 3: RED. Запустить каждый сценарий свежим субагентом без SKILL.md** (в промпте только задача, пути, доступ к ffmpeg и python). Записать в `baseline.md` выбор и рационализации дословно.
- [ ] **Step 4: Написать `SKILL.md`** под зафиксированные провалы: frontmatter (`name: speech-safe-cuts`, `description`: «Используй при любой резке видео с речью: когда пользователь говорит “не обрезай фразу”, “слышно обрывок”, “должно быть разборчиво”, “поставь слово целиком”, когда выбираешь границы сегмента с репликой, ставишь переход или удар рядом со словом или отчитываешься, что реплика целая»); тело: принцип (whisper говорит *какие* слова и *примерно где*, границы ставит только огибающая), поток из четырёх команд (`transcribe_words → envelope --word → check_cuts → verify_mix`), запреты, таблица рационализаций из `baseline.md`, красные флаги, формат отчёта пользователю (числа границ, onset, строка `check_cuts: problems 0`, результат `verify_mix`).
- [ ] **Step 5: GREEN. Перезапустить три сценария с SKILL.md в промпте** (текст скилла после строки `Base directory for this skill: <путь>`). Записать в `with-skill.md`. Ожидание: во всех трёх выбран вариант B/B/C и выполнены команды.
- [ ] **Step 6: REFACTOR.** Новые рационализации из шага 5 ⇒ строки в таблицу и красные флаги; перезапустить провалившийся сценарий. Повторять до трёх чистых прогонов.
- [ ] **Step 7: Проверить `wc -w skills/speech-safe-cuts/SKILL.md` ≤ 600** и `description` ≤ 500 знаков.
- [ ] **Step 8: Commit** `feat(speech-safe-cuts): skill with baseline-tested rules`.

### Task 9: SKILL.md и справочники для `match-grade`, `synth-score`, `trailer-build`; пример проекта «Землянка»

**Files:**
- Create: `skills/match-grade/SKILL.md`, `skills/synth-score/SKILL.md`, `skills/synth-score/references/instruments.md`, `skills/synth-score/references/recipes.md`, `skills/trailer-build/SKILL.md`, `skills/trailer-build/references/project-format.md`, `skills/trailer-build/references/dramaturgy.md`, `skills/trailer-build/references/example-trailer.json`

**Interfaces:**
- Consumes: все CLI задач 4–7.
- Produces: `example-trailer.json` = EDL и партитура v6 «Землянки» из `build.py` (`CLIPS`, `EDL`, `GAMMA/CONTRAST` как `grade: "gammas.txt"`, `score.layers`: дрон `s00→s20` рампа `[[s00,0.09],[s02,0.11],[s07,0.17],[s11,0.24],[s20,0.24]]` attack 3.0 release 0.4; дрон `s21→end` gain 0.16 attack 0.05 release 2.5; heartbeat 58 `s02→s07` gain 0.3; heartbeat 84 `s07→s10` gain 0.5; ticks `s07→s10` gain 0.18 (укоротить на 0.35 через `trim_end`); ostinato `s11→s19` gain 0.28 attack 0.2 release 0.3; riser как `hit` у `s19`; `hits {boom 0.75, braam 0.7, hit 0.55, titleboom 0.7, explosion 1.0}`; `s01.fade_in 1.5`; `s21 style painted bg black`; `s20.hit {type explosion, whistle_from s19}`), пути клипов относительные `../<файл>`.

- [ ] **Step 1: Написать `project-format.md`** (каждое поле схемы из задачи 2 с типом, умолчанием и одной строкой смысла; таблица «правка пользователя → какие поля → какие шаги перезапускать»).
- [ ] **Step 2: Написать `dramaturgy.md`** (структура тизера: тишина/карточка, акт ожидания, нарастание, атака короткими стыками с вспышками, обрыв в белое/тишину, слэм названия, «скоро»; типичные длительности; где удары, где дакинг; что делать с репликами: отсылка к `speech-safe-cuts`).
- [ ] **Step 3: Написать `instruments.md`** (таблица инструментов и событий: параметры, характер, где уместно) **и `recipes.md`** (нарастание к взрыву, слэм названия, сердцебиение двух темпов, тишина перед титром, дакинг под речь, цель −17 LUFS).
- [ ] **Step 4: Написать три SKILL.md** (на русском, ≤ 500 слов тела, `description` только триггеры с фразами пользователя):
  - `match-grade`: поток `probe_clips → solve_grade → preview_grid → смотреть сетку через Read → группы контраста при необходимости → render_graded по запросу`; критерий готовности (средняя яркость всех клипов в ±2 от цели); отдаваемые артефакты.
  - `synth-score`: поток `описать слои в score → build.py audio → audio_report → слушать нельзя, смотреть спектрограмму и LUFS → правки только через score/hit`; **REQUIRED SUB-SKILL:** `speech-safe-cuts` для всего, что рядом с репликами.
  - `trailer-build`: поток разведка → `speech-safe-cuts` (transcribe) и `match-grade` → `trailer.json` → `edl → cards → segments → film → assemble → audio → mux → preview` → `contact_sheet` и `check_cuts`, `verify_mix` → сдача (мастер, превью, таблица правок, допущения); чеклист перед сдачей; правило «правки только через trailer.json и перезапуск шагов»; ссылки на справочники и на `example-trailer.json` как образец.
- [ ] **Step 5: Собрать `example-trailer.json`** и проверить `build.py example-trailer.json edl` на копии в `/Users/leogodnik/Землянка/_сборка/trailer.json`. Ожидание: `total 66.42`, 25 сегментов.
- [ ] **Step 6: Commit** `docs: skills and references for trailer-build, match-grade, synth-score; example project`.

### Task 10: Приёмка: пересборка «Землянки» v6 плагином

**Files:**
- Create: `tests/acceptance/zemlyanka-v6.md`
- Modify (вне репозитория, новый файл): `/Users/leogodnik/Землянка/_сборка/trailer.json` (копия `example-trailer.json`; `build.py` «Землянки» не трогать; вывод в `_сборка/out/`, корневые мастера не перезаписывать)

- [ ] **Step 1: Выполнить `build.py trailer.json wav cards segments film assemble audio mux preview`** из `_сборка/`. Ожидание: без ошибок, `count_frames(out/Землянка_трейлер.mp4) == 1594`.
- [ ] **Step 2: `check_cuts.py trailer.json --words words.json`**. Ожидание: `problems: 0`. Если нет: править `trailer.json` (границы), записать расхождение с v6 в протокол, не менять пороги.
- [ ] **Step 3: `verify_mix.py trailer.json --transcribe`**. Ожидание: exit 0; транскрипция содержит «Тяните», «Гнойный буйволизм», «Пот и чай», «Покажи котлы».
- [ ] **Step 4: `contact_sheet.py trailer.json out/sheet.jpg`** и такая же раскадровка корневого мастера; посмотреть обе через Read; сверить кадры на 24, 45, 56.5, 58.4 с. Ожидание: тот же монтаж, те же титры, плёнка видна, цветность нейтральна.
- [ ] **Step 5: Записать протокол** в `zemlyanka-v6.md` (команды, числа, отличия).
- [ ] **Step 6: Commit** `test: acceptance protocol, Zemlyanka v6 rebuilt by the plugin`.

### Task 11: Сценарии применения для трёх технических скиллов

**Files:**
- Create: `tests/skills/trailer-build/scenario.md`, `protocol.md`; `tests/skills/match-grade/scenario.md`, `protocol.md`; `tests/skills/synth-score/scenario.md`, `protocol.md`
- Modify: соответствующие `SKILL.md` по находкам

- [ ] **Step 1: Написать сценарии**: `match-grade` («в папке X три клипа разной яркости, приведи к одному свету в нуаре, покажи до/после», клипы `c2, c4, c5` «Землянки» в scratchpad); `synth-score` («вот trailer.json без score, нужна саспенс-музыка: тихое начало, удар на карточке, нарастание к титру», mini-проект с реальными клипами); `trailer-build` («собери 20-секундный тизер из двух клипов с одной карточкой и названием», клипы `c2`, `c7`).
- [ ] **Step 2: Запустить каждый свежим субагентом** с текстом SKILL.md (после строки `Base directory for this skill:`) и без других подсказок. Записать в `protocol.md`: получился ли результат, какие команды не нашёл, где спросил бы автора.
- [ ] **Step 3: Поправить SKILL.md/справочники по находкам, перезапустить провалившиеся сценарии.** Ожидание: все три дают корректный артефакт (файл есть, длина совпадает с таймлайном, `check_cuts` 0 проблем там, где есть речь).
- [ ] **Step 4: Commit** `docs: skills refined after application scenarios`.

### Task 12: README, установка, память

**Files:**
- Modify: `README.md`, `.claude-plugin/plugin.json` (проверить версию), `/Users/leogodnik/.claude/projects/-Users-leogodnik---------/memory/zemlyanka-trailer-pipeline.md`, `MEMORY.md`

- [ ] **Step 1: Дописать README** (на русском: что умеет, четыре скилла одной строкой каждый, требования `ffmpeg`, `uv`, установка `/plugin marketplace add leogodnik/leo50s-montage` и `/plugin install leo50s-montage@leo50s`, затем `bash scripts/setup_env.sh`; пример `trailer.json` из 6 сегментов; три команды для старта; ссылка на пример «Землянки»).
- [ ] **Step 2: Установить плагин локально через маркетплейс GitHub** (`claude plugin marketplace add leogodnik/leo50s-montage && claude plugin install leo50s-montage@leo50s` или команды `/plugin` в интерактивной сессии; если CLI недоступен из скрипта, записать в README, что делает пользователь). Ожидание: в `~/.claude/plugins/installed_plugins.json` есть `leo50s-montage@leo50s`.
- [ ] **Step 3: Обновить память проекта «Землянка»**: пайплайн теперь в плагине, `trailer.json` рядом с `build.py`, правки делать плагином; добавить `[[leo50s-montage]]` и новую запись `reference` с URL репозитория; строка в `MEMORY.md`.
- [ ] **Step 4: Запустить полный `pytest -q`** и `pytest -m slow -q`. Ожидание: PASS.
- [ ] **Step 5: Commit и push** `docs: README, install, 0.1.0`; `git tag v0.1.0 && git push --tags`.
