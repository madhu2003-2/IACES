"""Start the hosted dashboard and its private, same-process Flask API."""
import os
from pathlib import Path
import re
import sys
import tempfile
import threading

from flask import abort, g, request
from waitress import create_server

ROOT = Path(__file__).resolve().parents[1]


def run(session_root):
    os.environ.update(
        DATABASE_ENABLED="true",
        DATABASE_BACKEND="sqlite",
        IACES_HOSTED_SESSION_DIR=str(session_root),
        FITPREDICT_MODEL_PATH=str(ROOT / "deploy/workout.joblib"),
        FITPREDICT_MANIFEST_PATH=str(ROOT / "deploy/workout_manifest.json"),
    )
    # Import after applying the hosted settings. The local CLI retains its own config.
    from iaces.api.app import create_app
    from iaces.ml.predict import load_model
    from streamlit.web import cli

    load_model()
    app = create_app()

    @app.before_request
    def use_session_history():
        session = request.headers.get("X-IACES-Session", "")
        if not re.fullmatch(r"session-[a-z0-9_]+", session):
            abort(403)
        folder = session_root / session
        if not folder.is_dir():
            abort(403)
        g.workout_database_path = folder / "history.sqlite3"

    server = create_server(app, host="127.0.0.1", port=0)
    os.environ["FITPREDICT_API_URL"] = f"http://127.0.0.1:{server.effective_port}"
    worker = threading.Thread(target=server.run, daemon=True)
    worker.start()
    try:
        sys.argv = [
            "streamlit", "run", str(ROOT / "frontend/streamlit_app.py"),
            f"--server.address={os.getenv('STREAMLIT_SERVER_ADDRESS', '0.0.0.0')}",
            f"--server.port={os.getenv('PORT', '8501')}",
            "--server.headless=true", "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false",
        ]
        cli.main()
    finally:
        server.task_dispatcher.shutdown()
        server.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="iaces-hosted-") as directory:
        run(Path(directory))
