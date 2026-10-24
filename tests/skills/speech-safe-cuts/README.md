# Тесты навыка speech-safe-cuts

Три сценария давления проверяют дисциплину навыка: границы речевого сегмента ставит огибающая звука, а не таймкоды whisper (whisper отстаёт примерно на 0,3 с), и «слово целое» доказывается проверкой микса (`verify_mix.py`), а не словами.
Контроллер запускает каждый файл из `scenarios/` свежим субагентом: сначала без текста SKILL.md (RED), потом с текстом навыка в промпте (GREEN).
Результаты записываются в `baseline.md` (без навыка) и `with-skill.md` (с навыком): выбранный вариант и рационализации дословно.
Тестовый проект лежит в scratchpad сессии (`$S`), в репозиторий не входит. Пересоздать:

    S=<scratchpad>/ssc-test; mkdir -p $S/clips $S/_сборка
    cp /Users/leogodnik/Землянка/gemini_generated_video_d6940d17.mp4 $S/clips/shout.mp4
    # trailer.json: clips {"shout":"../clips/shout.mp4"}, fps 24, size [1920,1080], look noir;
    # EDL: s00 black 1.0; s01 clip shout ss 1.68 d 0.67 ov 0.3 trans fade speech; s02 clip shout ss 3.24 d 1.3 ov 0.2 trans fadewhite hit; s03 card "СКОРО" 2.0; s04 black 1.0
    # words.json: {"shout": <список c6 из Землянка/_сборка/words.json>}
    ~/.cache/leo50s-montage/venv/bin/python skills/trailer-build/scripts/build.py $S/_сборка/trailer.json wav
