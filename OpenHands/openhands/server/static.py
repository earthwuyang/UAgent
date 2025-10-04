from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope, Receive, Send


class SPAStaticFiles(StaticFiles):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Skip WebSocket connections - they should be handled by WebSocket routes
        # StaticFiles only handles HTTP, so we reject WebSocket here
        if scope["type"] == "websocket":
            # Close the WebSocket connection immediately
            await send({
                "type": "websocket.close",
                "code": 1002,  # Protocol error
                "reason": "WebSocket not supported by static file handler"
            })
            return

        # Handle HTTP requests normally
        await super().__call__(scope, receive, send)

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except Exception:
            # FIXME: just making this HTTPException doesn't work for some reason
            return await super().get_response('index.html', scope)
