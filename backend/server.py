"""
FastAPI WebSocket backend.

Loads the model once at startup, then exposes /ws/generate: a client
connects, sends one JSON message with the prompt + generation params,
and receives a "step" event after every DiffusionEngine refinement step,
followed by a single "complete" (or "error") event.

    Run command: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import queue
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import DiffusionEngine
from model_loader import load_model, default_checkpoint_path

MODEL_PATH = os.environ.get("LLADA_MODEL_PATH", default_checkpoint_path())

# Tested on RTX4060 8GB vram, hence only one generation at a time.
generation_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_PATH} ...")
    model, tokenizer = load_model(MODEL_PATH)
    app.state.engine = DiffusionEngine(model, tokenizer)
    print("Model loaded, ready for connections.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerationRequest(BaseModel):
    prompt: str
    steps: int = 16
    gen_length: int = 32
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


@app.get("/")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


def _run_generation_in_thread(engine: DiffusionEngine, params: dict, result_queue: "queue.Queue"):
    try:
        for state in engine.generate_stream(**params):
            result_queue.put(("step", state))
        result_queue.put(("complete", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    await websocket.accept()

    try:
        payload = await websocket.receive_json()
        request = GenerationRequest(**payload)
    except Exception as e:  # noqa: BLE001 - bad JSON / failed validation
        await websocket.send_json({"type": "error", "message": f"Invalid request: {e}"})
        await websocket.close()
        return

    if generation_lock.locked():
        await websocket.send_json(
            {"type": "error", "message": "Another generation is already in progress. Try again shortly."}
        )
        await websocket.close()
        return

    async with generation_lock:
        engine: DiffusionEngine = websocket.app.state.engine
        result_queue: "queue.Queue" = queue.Queue()

        params = request.model_dump()
        params.pop("prompt")
        thread = threading.Thread(
            target=_run_generation_in_thread,
            args=(engine, {"prompt": request.prompt, **params}, result_queue),
            daemon=True,
        )
        thread.start()

        loop = asyncio.get_event_loop()
        try:
            while True:
                kind, value = await loop.run_in_executor(None, result_queue.get)

                if kind == "step":
                    await websocket.send_json(
                        {
                            "type": "step",
                            "step": value.step,
                            "total_steps": value.total_steps,
                            "tokens": [asdict(t) for t in value.tokens],
                        }
                    )
                elif kind == "complete":
                    await websocket.send_json({"type": "complete"})
                    break
                elif kind == "error":
                    await websocket.send_json({"type": "error", "message": value})
                    break
        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()