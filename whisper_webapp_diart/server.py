from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import websockets

app = FastAPI()

# Serve index.html
@app.get("/")
async def get():
    return HTMLResponse(content=open("index.html").read(), status_code=200)

# WebSocket Proxy
WS_SERVER_URL = "ws://127.0.0.1:7007"

@app.websocket("/ws")
async def websocket_proxy(websocket: WebSocket):
    """ Proxy WebSocket messages between frontend and backend WebSocket server """
    await websocket.accept()
    try:
        async with websockets.connect(WS_SERVER_URL) as ws_server:
            async def receive_from_client():
                """ Receive messages from frontend and send to backend WebSocket server """
                try:
                    while True:
                        data = await websocket.receive_text()
                        # print(f"Forwarding message to backend WS: {data}")
                        await ws_server.send(data)
                except WebSocketDisconnect:
                    print("Frontend WebSocket disconnected.")

            async def receive_from_server():
                """ Receive messages from backend WebSocket server and send to frontend """
                try:
                    while True:
                        data = await ws_server.recv()
                        print(f"Received from backend WS: {data}")
                        await websocket.send_text(data)
                except websockets.exceptions.ConnectionClosed:
                    print("Backend WebSocket disconnected.")

            # Run both communication tasks concurrently
            await asyncio.gather(receive_from_client(), receive_from_server())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
