from datetime import datetime
from pathlib import Path
import subprocess
import sys
import traceback
from threading import Lock, Thread

import imageio_ffmpeg
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_socketio import SocketIO, emit

try:
    from .database import create_analysis, initialize, list_analyses, update_analysis
except ImportError:
    from database import create_analysis, initialize, list_analyses, update_analysis

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VIDEO_ROOT = ROOT / "01_videos_brutos"
WEB_VIDEO_ROOT = Path(__file__).resolve().parent / "cache" / "videos"
ANNOTATED_VIDEO_ROOT = Path(__file__).resolve().parent / "cache" / "annotated"

app = Flask(__name__)
app.config["SECRET_KEY"] = "vigil-ia-mvp-change-me"
socketio = SocketIO(app, cors_allowed_origins="*")
processing_lock = Lock()
conversion_lock = Lock()
initialize()

USERS = {"admin": "vigil-ia"}


def logged_in():
    return "user" in session


def video_catalog():
    videos = []
    if not VIDEO_ROOT.exists():
        return videos
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    for path in VIDEO_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            relative = path.relative_to(VIDEO_ROOT).as_posix()
            parts = path.relative_to(ROOT).parts
            category = parts[1] if len(parts) > 1 else "outros"
            videos.append({
                "id": relative,
                "name": path.name,
                "category": category,
                "url": url_for("video_file", relative_path=relative),
                "annotated_url": url_for("annotated_video_file", relative_path=relative),
            })
    return sorted(videos, key=lambda item: item["name"].lower())


def navegador_video_path(source):
    relative = source.relative_to(VIDEO_ROOT).with_suffix(".mp4")
    target = WEB_VIDEO_ROOT / relative
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.mp4")
    with conversion_lock:
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i", str(source),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "14",
            "-pix_fmt", "yuv420p",
            "-an",
            "-movflags", "+faststart",
            str(temporary),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Falha ao converter vídeo: {result.stderr[-500:]}")
        temporary.replace(target)
    return target


def video_anotado_path(source):
    relative = source.relative_to(VIDEO_ROOT).with_suffix(".mp4")
    target = ANNOTATED_VIDEO_ROOT / relative
    with conversion_lock:
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target

        try:
            from .annotator import gerar_video_marcado
        except ImportError:
            from annotator import gerar_video_marcado

        target.parent.mkdir(parents=True, exist_ok=True)
        raw_target = target.with_suffix(".raw.mp4")
        gerar_video_marcado(source, raw_target)
        converted = target.with_suffix(".partial.mp4")
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw_target),
            "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
            "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", str(converted),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        raw_target.unlink(missing_ok=True)
        if result.returncode != 0:
            converted.unlink(missing_ok=True)
            raise RuntimeError(f"Falha ao gerar vídeo anotado: {result.stderr[-500:]}")
        converted.replace(target)
    return target


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if USERS.get(username) == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Credenciais inválidas")
    return render_template("login.html")


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["user"])


@app.get("/media/<path:relative_path>")
def video_file(relative_path):
    return serve_video(relative_path, annotated=False)


@app.get("/media-annotated/<path:relative_path>")
def annotated_video_file(relative_path):
    return serve_video(relative_path, annotated=True)


def serve_video(relative_path, annotated):
    if not logged_in():
        return redirect(url_for("login"))
    requested = (VIDEO_ROOT / relative_path).resolve()
    if VIDEO_ROOT.resolve() not in requested.parents or not requested.is_file():
        return jsonify({"error": "Vídeo não encontrado"}), 404
    try:
        web_video = video_anotado_path(requested) if annotated else navegador_video_path(requested)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 500
    return send_file(
        web_video,
        conditional=True,
        etag=True,
        max_age=0,
        mimetype="video/mp4" if requested.suffix.lower() == ".mp4" else None,
    )


def processar_analise(analysis_id, video):
    try:
        from classificar_video import classificar_video

        with processing_lock:
            classification, risks = classificar_video(
                str((VIDEO_ROOT / video["id"]).resolve()),
                limiar=42.5,
                mostrar=False,
                proporcao_suspeita=0.20,
            )
        risks = list(risks)
        average = float(sum(float(risk) for risk in risks) / len(risks)) if risks else 0.0
        suspicious = float(sum(risk >= 42.5 for risk in risks) / len(risks)) if risks else 0.0
        update_analysis(
            analysis_id,
            classification=classification,
            risk=float(round(average, 2)),
            frames=len(risks),
            suspicious_frames=round(suspicious, 4),
            status="ANALISADO",
            updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
    except Exception:
        traceback.print_exc()
        update_analysis(
            analysis_id,
            classification="ERRO",
            status="ERRO",
            updated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
    analysis = next(item for item in list_analyses() if item["id"] == analysis_id)
    socketio.emit("analysis_update", analysis)


@app.get("/api/videos")
def videos_api():
    if not logged_in():
        return jsonify({"error": "Não autenticado"}), 401
    return jsonify(video_catalog())


@app.get("/api/analyses")
def analyses_api():
    if not logged_in():
        return jsonify({"error": "Não autenticado"}), 401
    return jsonify(list_analyses())


@app.post("/api/analyze")
def analyze_api():
    if not logged_in():
        return jsonify({"error": "Não autenticado"}), 401
    payload = request.get_json(silent=True) or {}
    video_id = payload.get("video_id")
    video = next((item for item in video_catalog() if item["id"] == video_id), None)
    if video is None:
        return jsonify({"error": "Vídeo não encontrado"}), 404
    analysis_id = f"job-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    analysis = create_analysis(
        analysis_id,
        video["name"],
        video["category"],
        analyzed_by=session["user"],
    )
    socketio.emit("analysis_update", analysis)
    Thread(target=processar_analise, args=(analysis_id, video), daemon=True).start()
    return jsonify(analysis), 202


@socketio.on("connect")
def connected():
    if logged_in():
        emit("system_status", {"status": "online", "message": "Vigil-IA conectado"})


if __name__ == "__main__":
    socketio.run(app, debug=False, use_reloader=False, host="127.0.0.1", port=5000, allow_unsafe_werkzeug=True)
