import asyncio
import websockets
from aiohttp import web
import json
from pathlib import Path
import threading
from typing import List, Set

class State:
    def __init__(self):
        self.ui_sessions: Set[websockets.WebSocketServerProtocol] = set()
        self.backend_sessions: Set[websockets.WebSocketServerProtocol] = set()
        self.namespaces: List[int] = []
        self.lock = threading.Lock()
        self.scenario = ""

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    path = request.path
    state = request.app['state']

    try:
        is_ui = path == "/ui"

        with state.lock:
            if is_ui:
                state.ui_sessions.add(ws)
                print("UI client connected")
            else:  # backend
                state.backend_sessions.add(ws)
                namespace = len(state.namespaces)
                state.namespaces.append(namespace)
                print(f"Backend connected: {namespace}")
                handshake = {
                    "channel": "handshake",
                    "data": {
                        "namespace": str(namespace)
                    }
                }
                await ws.send_json(handshake)

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    parsed = json.loads(msg.data)

                    with state.lock:
                        if is_ui:
                            for backend in state.backend_sessions:
                                try:
                                    await backend.send_str(msg.data)
                                except Exception:
                                    continue
                        else:
                            print(f"Received message: {msg.data}")
                            for ui in state.ui_sessions:
                                try:
                                    await ui.send_str(msg.data)
                                except Exception:
                                    continue
                except json.JSONDecodeError:
                    print(f"Invalid JSON message received: {msg.data}")
                    continue
            elif msg.type == web.WSMsgType.ERROR:
                print(f'WebSocket connection closed with exception {ws.exception()}')

    finally:
        with state.lock:
            if is_ui:
                state.ui_sessions.remove(ws)
                print("UI client disconnected")
            else:
                state.backend_sessions.remove(ws)
                print("Backend client disconnected")

    return ws

async def handle_static(request):
    static_path = request.app['static_path']
    path = request.path

    if path == "/" or path == "":
        path = "/index.html"

    file_path = Path(static_path) / path.lstrip('/')

    if not file_path.exists():
        return web.Response(text="File not found", status=404)

    content_type = 'text/plain'
    if file_path.suffix == '.html':
        content_type = 'text/html'
    elif file_path.suffix == '.js':
        content_type = 'application/javascript'
    elif file_path.suffix == '.css':
        content_type = 'text/css'
    elif file_path.suffix == '.wasm':
        content_type = 'application/wasm'

    return web.FileResponse(file_path, headers={'Content-Type': content_type})

async def handle_scenario(request):
    state = request.app['state']
    return web.Response(text=state.scenario, content_type='text/html')

def start_server(static_path: str = "./static", port: int = 8080, scenario: str = ""):
    state = State()
    state.scenario = scenario

    app = web.Application()
    app['state'] = state
    app['static_path'] = static_path

    app.router.add_get('/ui', websocket_handler)
    app.router.add_get('/backend', websocket_handler)
    app.router.add_get('/scenario', handle_scenario)
    app.router.add_get('/{tail:.*}', handle_static)

    print(f"Server starting at http://localhost:{port}")
    print(f"WebSocket endpoints at ws://localhost:{port}/ui and ws://localhost:{port}/backend")

    web.run_app(app, port=port)

if __name__ == "__main__":
    start_server(
        static_path="./static/ui_server/generic",
        port=8080,
        scenario="generic"
    )

