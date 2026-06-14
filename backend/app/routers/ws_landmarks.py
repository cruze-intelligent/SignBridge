from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
from app.inference import predict_sequence
# If you have your persistence module for saving .npy files, import it here
# from app.persistence import save_sequence_to_disk 

router = APIRouter()

@router.websocket("/ws/landmarks")
async def websocket_landmarks(websocket: WebSocket):
    await websocket.accept()

    # Immediately confirm the bridge is open so the frontend knows
    # the WSS handshake completed (critical when proxied via Railway).
    await websocket.send_json({
        "status": "connected",
        "detail": "SignBridge backend is listening...",
    })

    sequence_buffer = []
    batch_count     = 0   # number of complete 30-frame batches evaluated this session
    
    # Retrieve the warmed-up model and label map from FastAPI's global state
    model     = websocket.app.state.model
    label_map = getattr(websocket.app.state, 'label_map', [])
    
    try:
        while True:
            # Await incoming JSON payload from the Vanilla JS client
            data = await websocket.receive_json()
            
            # 1. Handle UI Controls (Start/Stop/Save)
            if "action" in data:
                action = data["action"]
                if action in ["save", "end"]:
                    # Persist trailing frames if needed:
                    # save_sequence_to_disk(sequence_buffer)
                    sequence_buffer = []
                    await websocket.send_json({
                        "status": "saved" if action == "save" else "session_ended"
                    })
                    if action == "end":
                        break  # Close the connection cleanly
                continue
            
            # 2. Handle Live Data Streaming
            frame = data.get("frame")
            if frame is not None and len(frame) == 225:
                sequence_buffer.append(frame)
                print(
                    f"[FRAME] {len(sequence_buffer):02d}/30  "
                    f"(batch #{batch_count + 1})"
                )
                
            # 3. Inference Trigger — fires on every complete 30-frame batch
            #
            # KEY DESIGN DECISION: We always flush the buffer after inference,
            # whether or not a sign was detected.  This guarantees each inference
            # call receives a temporally coherent sequence that exactly matches
            # the 30-frame windows the model was trained on.
            #
            # The old sliding-window approach (pop(0)) created "chimera" windows
            # spanning two different gestures, which the model had never seen
            # during training, causing random false predictions.
            if len(sequence_buffer) == 30:
                batch_count += 1

                if model is not None:
                    sequence_array = np.array(sequence_buffer, dtype=np.float32)
                    
                    try:
                        prediction = predict_sequence(model, sequence_array, label_map)
                        
                        if prediction != "...":
                            await websocket.send_json({
                                "status": "translated",
                                "text": prediction
                            })
                            print(f"[OK] TRANSLATED (batch #{batch_count}): {prediction}")
                        else:
                            print(
                                f"[--] Batch #{batch_count}: no confident prediction - "
                                "buffer cleared, waiting for next gesture"
                            )
                            
                    except Exception as e:
                        print(f"[ERR] INFERENCE CRASH (batch #{batch_count}): {e}")
                        await websocket.send_json({
                            "status": "error", 
                            "detail": f"Inference failed: {str(e)}"
                        })
                
                # Always clear — never slide across gesture boundaries
                sequence_buffer = []

    except WebSocketDisconnect:
        # Client disconnected cleanly — no action needed
        print("[INFO] Client disconnected from /ws/landmarks")

    except Exception as e:
        # Unexpected error on the outer loop — log and attempt graceful close
        print(f"[CRASH] WebSocket session crashed: {e}")
        try:
            await websocket.send_json({"status": "error", "detail": str(e)})
            await websocket.close()
        except Exception:
            pass  # Socket may already be closed

    finally:
        print(f"[END] Session ended. Batches evaluated: {batch_count}. Buffer cleared.")