# Сценарии применения технических скиллов

Сценарий проверяет, справится ли свежий агент, у которого есть только текст `SKILL.md`.
Контролёр даёт агенту `scenario.md` целиком, перед ним `Base directory for this skill: …` и текст навыка.
Фикстуры лежат в scratchpad (`app-tests/{grade,score,teaser}`) и не коммитятся. Пересборка:
- клипы копируются из `/Users/leogodnik/Землянка/gemini_generated_video_*.mp4` (c2 `4471ee0d`, c4 `7d3482cb`, c5 `aee2631e`, c7 `f4c6042c`) в `<фикстура>/clips/<алиас>.mp4`;
- `grade`: c2, c4, c5, без проекта; `score`: c2, c7, `_сборка/trailer.json` без `score`, слова из `_сборка/words.json`;
- `teaser`: c2, c7 и только `words.json`;
- в каждой папке `README-test.md` (одноразовая фикстура).
Протокол прогона пишется в `protocol.md` рядом с каждым `scenario.md`.
