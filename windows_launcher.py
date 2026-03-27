from __future__ import annotations

import logging
import socket
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
LOG_FILE = Path.cwd() / "people-counter-launcher.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("people_counter.launcher")


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _open_browser_when_ready(timeout_seconds: int = 30) -> None:
    started = time.time()
    while time.time() - started < timeout_seconds:
        if _port_open(HOST, PORT):
            webbrowser.open(URL)
            return
        time.sleep(0.2)


def main() -> None:
    LOGGER.info("Launcher start host=%s port=%s", HOST, PORT)
    # If server is already running, just open browser.
    if _port_open(HOST, PORT):
        LOGGER.info("Server already running, opening browser: %s", URL)
        webbrowser.open(URL)
        return

    thread = threading.Thread(target=_open_browser_when_ready, daemon=True)
    thread.start()

    try:
        uvicorn.run("web.app:app", host=HOST, port=PORT, reload=False, log_level="info")
    except Exception:
        LOGGER.exception("Launcher failed")
        raise


if __name__ == "__main__":
    main()
