from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
from app.inference import predict_frame
# from app.persistence import save_sequence_to_disk

router = APIRouter()

# For static model, we process 1 frame at a time.
WINDOW_SIZE: int = 1


@router.websocket("/ws/landmarks")
async def websocket_landmarks(websocket: WebSocket):
    await websocket.accept()

    # Immediately confirm the bridge is open so the frontend knows
    # the WSS handshake completed (critical when proxied via Railway).
    await websocket.send_json({
        "status": "connected",
        "detail": "SignBridge backend is listening...",
    })

    # -----------------------------------------------------------------------
    # Indexed frame slots  (dict[int, list[float]])
    #
    # The frontend can send frames in any order when multiple async tasks race
    # over the WebSocket.  Keying by `frame_index` (0-based, 0-29) lets us
    # reassemble the correct temporal sequence regardless of arrival order.
    #
    # If the frontend sends frames WITHOUT a `frame_index` field (legacy mode)
    # we fall back to a simple append list so older clients keep working.
    # -----------------------------------------------------------------------
    frame_slots: dict[int, list] = {}   # indexed delivery (new clients)
    frame_list:  list[list]      = []   # ordered append  (legacy clients)
    batch_count: int             = 0

    # Retrieve the warmed-up model and label map from FastAPI's global state
    model     = websocket.app.state.model
    label_map = getattr(websocket.app.state, "label_map", [])

    try:
        while True:
            data = await websocket.receive_json()

            # ── 1. UI Controls (start / stop / save) ──────────────────────
            if "action" in data:
                action = data["action"]
                if action in ("save", "end"):
                    frame_slots.clear()
                    frame_list.clear()
                    await websocket.send_json({
                        "status": "saved" if action == "save" else "session_ended"
                    })
                    if action == "end":
                        await websocket.close(code=1000)
                        break
                continue

            # ── 2. Incoming landmark frame ─────────────────────────────────
            frame = data.get("frame")
            if frame is None or len(frame) != 225:
                continue   # malformed payload — skip silently

            frame_array = np.array(frame, dtype=np.float32)
            ready = True

            # ── 3. Inference — fires immediately for static model ──
            if not ready:
                continue

            batch_count += 1

            if model is None:
                # Model not loaded — send heartbeat to keep connection alive
                await websocket.send_json({"status": "processing"})
                print(f"[--] Batch #{batch_count}: no model loaded")
                continue

            try:
                prediction = predict_frame(model, frame_array, label_map)

                if prediction != "...":
                    await websocket.send_json({
                        "status": "translated",
                        "text": prediction,
                    })
                    print(f"[OK] TRANSLATED (batch #{batch_count}): {prediction}")
                else:
                    # Heartbeat: resets Railway's 60-second proxy idle-timeout
                    await websocket.send_json({"status": "processing"})
                    print(f"[--] Batch #{batch_count}: below confidence threshold")

            except Exception as exc:
                print(f"[ERR] INFERENCE CRASH (batch #{batch_count}): {exc}")
                await websocket.send_json({
                    "status": "error",
                    "detail": f"Inference failed: {exc}",
                })
                # Heartbeat: resets Railway's 60-second proxy idle-timeout on error
                await websocket.send_json({"status": "processing"})

    except WebSocketDisconnect:
        print("[INFO] Client disconnected from /ws/landmarks")

    except Exception as exc:
        print(f"[CRASH] WebSocket session crashed: {exc}")
        try:
            await websocket.send_json({"status": "error", "detail": str(exc)})
            await websocket.close()
        except Exception:
            pass

    finally:
        print(f"[END] Session ended. Batches evaluated: {batch_count}.")