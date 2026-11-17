"""Production Runner for FastAPI Chatbot.
Handles dynamic PORT binding across Railway, Render, Docker, and local environments.
Works seamlessly whether executed from repo root or backend subdirectory.
"""
import os
import sys

# Ensure backend directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, "backend", "app")):
    sys.path.insert(0, os.path.join(current_dir, "backend"))
elif os.path.exists(os.path.join(current_dir, "app")):
    sys.path.insert(0, current_dir)

import uvicorn

if __name__ == "__main__":
    port_env = os.environ.get("PORT", "8080")
    try:
        port = int(port_env)
    except ValueError:
        port = 8080

    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting uvicorn on {host}:{port} (PORT={port_env})...", flush=True)
    uvicorn.run("app.main:app", host=host, port=port)
