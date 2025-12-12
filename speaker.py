# speaker.py
import time
import subprocess

class Speaker:
    def __init__(self, cooldown_sec: int = 45):
        self.cooldown_sec = cooldown_sec
        self._last_ts = 0.0

    def _say(self, voice: str, text: str, speed: int = 165):
        # espeak-ng voices: uk, ru
        subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), text], check=False)

    def greet_uk_ru(self):
        now = time.time()
        if now - self._last_ts < self.cooldown_sec:
            return
        self._last_ts = now
        self._say("uk", "Привіт! Я тебе бачу.")
        time.sleep(0.2)
        self._say("ru", "Привет! Я тебя вижу.")
