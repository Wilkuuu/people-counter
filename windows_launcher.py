from __future__ import annotations

import socket
import threading
import time
import webbrowser

import uvicorn


HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


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
    # If server is already running, just open browser.
    if _port_open(HOST, PORT):
        webbrowser.open(URL)
        return

    thread = threading.Thread(target=_open_browser_when_ready, daemon=True)
    thread.start()

    uvicorn.run("web.app:app", host=HOST, port=PORT, reload=False, log_level="info")


if __name__ == "__main__":
    main()
