from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
from app.inference import predict_sequence
# If you have your persistence module for saving .npy files, import it here
# from app.persistence import save_sequence_to_disk 

router = APIRouter()

@router.websocket("/ws/landmarks")
async def websocket_landmarks(websocket: WebSocket):
    await websocket.accept()
    
    sequence_buffer = []
    
    # Retrieve the warmed-up model and label map from FastAPI's global state
    model = websocket.app.state.model
    label_map = getattr(websocket.app.state, 'label_map', [])
    
    try:
        while True:
            # Await incoming JSON payload from the Vanilla JS client
            data = await websocket.receive_json()
            
            # 1. Handle UI Controls (Start/Stop/Save)
            if "action" in data:
                action = data["action"]
                if action in ["save", "end"]:
                    # If you want to continue recording background data, call save_sequence_to_disk(sequence_buffer) here
                    sequence_buffer = []  # Flush the buffer
                    await websocket.send_json({
                        "status": "saved" if action == "save" else "session_ended"
                    })
                    if action == "end":
                        break  # Close the connection cleanly
                continue
            
            # 2. Handle Live Data Streaming
            frame = data.get("frame")
            if frame is not None:
                # --- WIRETAP: Print exactly what is arriving ---
                print(f"📡 Incoming frame -> Length: {len(frame)} | Buffer: {len(sequence_buffer) + 1}/30")

                if len(frame) == 225:
                    sequence_buffer.append(frame)
                
            # 3. The Inference Trigger (30-Frame Window)
            if len(sequence_buffer) == 30:
                if model is not None:
                    sequence_array = np.array(sequence_buffer, dtype=np.float32)
                    
                    try:
                        # Pass to the inference engine
                        prediction = predict_sequence(model, sequence_array, label_map)
                        
                        # If a confident prediction is made, blast it back to the UI
                        if prediction != "...":
                            await websocket.send_json({
                                "status": "translated",
                                "text": prediction
                            })
                            print(f"TRANSLATED: {prediction}")
                            
                    except Exception as e:
                        print(f"🔥 INFERENCE CRASH: {e}")
                        await websocket.send_json({
                            "status": "error", 
                            "detail": f"Inference failed: {str(e)}"
                        })
                
                # Clear the buffer to start accumulating the next 30 frames
                sequence_buffer = []  
    except WebSocketDisconnect:
        print("WebSocket Client Disconnected.")