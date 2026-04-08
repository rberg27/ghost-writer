"""
training/server.py

WebSocket server that bridges Arduino serial data to a browser-based
training data collection UI.

Usage:
    python -m training.server
    python training/server.py
    python training/server.py --port /dev/cu.usbmodem2101 --http-port 8765
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

# Add parent dir so we can import serial_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serial_utils import find_arduino_port, open_serial, parse_line
from training.dataset import append_sample, make_sample, get_stats, load_samples, delete_sample

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "training_data" / "samples.jsonl"


class SerialBridge:
    """Reads serial data in a background thread, buffers for streaming and recording."""

    def __init__(self, port, loop):
        self.port = port
        self.loop = loop
        self.connected = False
        self.running = True
        self.sample_rate = 0.0

        # Live streaming buffer (last 200 points for the chart)
        self.ring = deque(maxlen=200)

        # Word recording state
        self.lock = threading.Lock()
        self.recording = False
        self.recording_buffer = []
        self.recording_start = 0.0

        # Session recording state (continuous, captures gaps between words)
        self.session_active = False
        self.session_buffer = []
        self.session_start = 0.0
        self.session_wall_clock = ""

        # Subscribers (async queues for each connected WS client)
        self.subscribers = set()

        self.thread = threading.Thread(target=self._serial_loop, daemon=True)
        self.thread.start()

    def _serial_loop(self):
        ser = None
        rate_samples = 0
        rate_start = time.time()

        while self.running:
            # Connect / reconnect
            if ser is None:
                try:
                    port = self.port or find_arduino_port()
                    if port is None:
                        self._set_connected(False)
                        time.sleep(2)
                        continue
                    ser = open_serial(port)
                    self.port = port
                    self._set_connected(True)
                    rate_samples = 0
                    rate_start = time.time()
                except Exception:
                    self._set_connected(False)
                    time.sleep(2)
                    continue

            # Read data
            try:
                raw = ser.readline()
                if not raw:
                    continue
                parsed = parse_line(raw)
                if parsed is None:
                    continue

                x, y, z = parsed
                t = time.time()

                # Sample rate tracking
                rate_samples += 1
                elapsed = t - rate_start
                if elapsed >= 2.0:
                    self.sample_rate = round(rate_samples / elapsed, 1)
                    rate_samples = 0
                    rate_start = t

                # Ring buffer for live chart
                self.ring.append((x, y, z, t))

                # Recording buffers
                with self.lock:
                    if self.recording:
                        self.recording_buffer.append((x, y, z, t))
                    if self.session_active:
                        self.session_buffer.append((x, y, z, t))

                # Push to subscribers
                if self.loop is not None and self.subscribers:
                    msg = json.dumps({"type": "accel", "x": x, "y": y, "z": z, "t": t})
                    for q in list(self.subscribers):
                        try:
                            self.loop.call_soon_threadsafe(q.put_nowait, msg)
                        except (asyncio.QueueFull, RuntimeError):
                            pass

            except Exception:
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                self._set_connected(False)
                time.sleep(1)

    def _set_connected(self, val):
        self.connected = val
        msg = json.dumps({
            "type": "status",
            "connected": val,
            "port": self.port or "",
            "sample_rate_hz": self.sample_rate,
        })
        if self.loop is not None:
            for q in list(self.subscribers):
                try:
                    self.loop.call_soon_threadsafe(q.put_nowait, msg)
                except (asyncio.QueueFull, RuntimeError):
                    pass

    def subscribe(self):
        q = asyncio.Queue(maxsize=100)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q):
        self.subscribers.discard(q)

    def start_recording(self):
        with self.lock:
            self.recording = True
            self.recording_buffer = []
            self.recording_start = time.time()
            self.recording_wall_clock = datetime.now(timezone.utc).isoformat()

    def stop_recording(self):
        with self.lock:
            self.recording = False
            buf = list(self.recording_buffer)
            start = self.recording_start
            wall_clock = self.recording_wall_clock
        return buf, start, wall_clock

    def start_session(self):
        with self.lock:
            self.session_active = True
            self.session_buffer = []
            self.session_start = time.time()
            self.session_wall_clock = datetime.now(timezone.utc).isoformat()

    def stop_session(self):
        with self.lock:
            self.session_active = False
            buf = list(self.session_buffer)
            start = self.session_start
            wall_clock = self.session_wall_clock
            self.session_buffer = []
        return buf, start, wall_clock

    def stop(self):
        self.running = False


def build_app(bridge, dataset_path):
    app = web.Application()

    async def on_startup(app):
        # Capture the REAL running event loop so the serial thread
        # can push data to WebSocket clients via call_soon_threadsafe.
        bridge.loop = asyncio.get_running_loop()

    app.on_startup.append(on_startup)

    async def index(request):
        return web.FileResponse(STATIC_DIR / "index.html")

    async def static_handler(request):
        name = request.match_info["filename"]
        path = STATIC_DIR / name
        if path.exists() and path.is_file():
            return web.FileResponse(path)
        raise web.HTTPNotFound()

    async def api_stats(request):
        stats = get_stats(str(dataset_path))
        return web.json_response(stats)

    async def api_samples(request):
        samples = load_samples(str(dataset_path))
        # Return in reverse chronological order, paginated
        samples.reverse()
        offset = int(request.query.get("offset", 0))
        limit = int(request.query.get("limit", 50))
        page = samples[offset:offset + limit]
        # Strip the heavy sample data for the list view
        lightweight = []
        for s in page:
            lightweight.append({
                "id": s["id"],
                "word": s["word"],
                "duration_s": s.get("duration_s", 0),
                "num_samples": s.get("num_samples", 0),
                "created_at": s.get("created_at", ""),
            })
        return web.json_response({"samples": lightweight, "total": len(samples)})

    async def api_download(request):
        path = Path(dataset_path)
        if not path.exists():
            raise web.HTTPNotFound(text="No dataset file yet")
        return web.FileResponse(
            path,
            headers={"Content-Disposition": f'attachment; filename="samples.jsonl"'},
        )

    async def api_delete_sample(request):
        sample_id = request.match_info["id"]
        found = delete_sample(str(dataset_path), sample_id)
        if found:
            return web.json_response({"deleted": True})
        raise web.HTTPNotFound()

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        q = bridge.subscribe()
        pending_samples = {}
        # Per-session JSONL path (None = no active session)
        session_jsonl = None

        # Send initial status
        await ws.send_str(json.dumps({
            "type": "status",
            "connected": bridge.connected,
            "port": bridge.port or "",
            "sample_rate_hz": bridge.sample_rate,
        }))

        # Send current stats (from default file for initial count)
        stats = get_stats(str(dataset_path))
        await ws.send_str(json.dumps({"type": "stats", **stats}))

        async def pump_serial():
            """Forward serial data from queue to WebSocket."""
            try:
                while True:
                    msg = await q.get()
                    if ws.closed:
                        break
                    await ws.send_str(msg)
            except Exception:
                pass

        pump_task = asyncio.create_task(pump_serial())

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    cmd = data.get("type")

                    if cmd == "start_recording":
                        bridge.start_recording()
                        await ws.send_str(json.dumps({"type": "recording_started"}))

                    elif cmd == "stop_recording":
                        buf, start, wall_clock = bridge.stop_recording()
                        sample_id = str(uuid.uuid4())
                        samples_list = [[x, y, z] for x, y, z, t in buf]
                        timestamps = [round(t - start, 4) for x, y, z, t in buf]
                        duration = timestamps[-1] if timestamps else 0.0
                        pending_samples[sample_id] = (samples_list, timestamps, wall_clock)
                        await ws.send_str(json.dumps({
                            "type": "recording_stopped",
                            "sample_id": sample_id,
                            "samples": samples_list,
                            "timestamps": timestamps,
                            "duration_s": round(duration, 3),
                            "num_samples": len(samples_list),
                            "recorded_at": wall_clock,
                        }))

                    elif cmd == "save_sample":
                        sid = data["sample_id"]
                        word = data["word"].strip()
                        if sid in pending_samples and word:
                            samples_list, timestamps, wall_clock = pending_samples.pop(sid)
                            audio_dir = dataset_path.parent / "audio"
                            audio_file_path = audio_dir / f"{sid}.wav"
                            audio_rel = f"audio/{sid}.wav" if audio_file_path.exists() else None
                            sample = make_sample(word, samples_list, timestamps,
                                                 recorded_at=wall_clock,
                                                 audio_file=audio_rel)
                            sample["id"] = sid
                            if "line" in data:
                                sample["line"] = data["line"]
                            # Write to session JSONL if active, else default
                            target = session_jsonl or str(dataset_path)
                            append_sample(target, sample)
                            stats = get_stats(target)
                            await ws.send_str(json.dumps({
                                "type": "sample_saved",
                                "sample_id": sid,
                                **stats,
                            }))

                    elif cmd == "discard_sample":
                        sid = data.get("sample_id", "")
                        pending_samples.pop(sid, None)
                        await ws.send_str(json.dumps({"type": "sample_discarded"}))

                    elif cmd == "start_session":
                        bridge.start_session()
                        # Create a new session JSONL + CSV
                        wall_clock = bridge.session_wall_clock
                        session_id = str(uuid.uuid4())[:8]
                        ts_str = wall_clock[:19].replace(":", "").replace("-", "").replace("T", "_")
                        sessions_dir = dataset_path.parent / "sessions"
                        sessions_dir.mkdir(parents=True, exist_ok=True)
                        session_base = f"session_{ts_str}_{session_id}"
                        session_jsonl = str(sessions_dir / f"{session_base}.jsonl")
                        await ws.send_str(json.dumps({
                            "type": "session_started",
                            "session_file": f"{session_base}.jsonl",
                        }))

                    elif cmd == "stop_session":
                        buf, start, wall_clock = bridge.stop_session()
                        # Save continuous accel stream as CSV
                        import csv as csv_mod
                        csv_path = Path(session_jsonl).with_suffix(".csv") if session_jsonl else None
                        if csv_path and buf:
                            with open(csv_path, "w", newline="") as f:
                                w = csv_mod.writer(f)
                                w.writerow(["elapsed_s", "x_g", "y_g", "z_g"])
                                for x, y, z, t in buf:
                                    w.writerow([f"{t - start:.4f}", f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])
                        duration = (buf[-1][3] - start) if buf else 0.0
                        session_name = Path(session_jsonl).stem if session_jsonl else ""
                        session_jsonl = None  # close session
                        await ws.send_str(json.dumps({
                            "type": "session_stopped",
                            "file": session_name,
                            "num_samples": len(buf),
                            "duration_s": round(duration, 2),
                        }))

                    elif cmd == "get_stats":
                        target = session_jsonl or str(dataset_path)
                        stats = get_stats(target)
                        await ws.send_str(json.dumps({"type": "stats", **stats}))

                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            pump_task.cancel()
            bridge.unsubscribe(q)

        return ws

    app.router.add_get("/", index)
    app.router.add_get("/static/{filename}", static_handler)
    app.router.add_get("/ws", websocket_handler)
    async def api_upload_audio(request):
        """Receive a .wav file for a sample ID."""
        sample_id = request.match_info["id"]
        audio_dir = dataset_path.parent / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{sample_id}.wav"
        # Read the raw body (WAV bytes)
        body = await request.read()
        with open(audio_path, "wb") as f:
            f.write(body)
        return web.json_response({"saved": True, "path": str(audio_path)})

    app.router.add_get("/api/stats", api_stats)
    app.router.add_get("/api/samples", api_samples)
    app.router.add_get("/api/download", api_download)
    app.router.add_post("/api/samples/{id}/audio", api_upload_audio)
    app.router.add_delete("/api/samples/{id}", api_delete_sample)

    return app


def main():
    parser = argparse.ArgumentParser(description="Ghost Writer Training Data Collector")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--http-port", type=int, default=8765,
                        help="HTTP/WebSocket port (default: 8765)")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                        help="Path to JSONL dataset file")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass None for loop — the real loop is captured in on_startup
    bridge = SerialBridge(args.port, None)
    app = build_app(bridge, dataset_path)

    print(f"Starting Ghost Writer Training Server...")
    print(f"  Serial: {args.port or 'auto-detect'}")
    print(f"  Dataset: {dataset_path}")
    print(f"  UI: http://localhost:{args.http_port}")
    print()

    try:
        web.run_app(app, host="localhost", port=args.http_port, print=None)
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
