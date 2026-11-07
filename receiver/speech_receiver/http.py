import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from urllib.parse import parse_qs

from prometheus_client import generate_latest
from starlette.requests import Request

from .engine import Engine, Rejected
from .journal import KafkaJournal


async def reply(
    send, status, body, content_type="application/json", execution="not-started"
):
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", content_type.encode()),
                (b"x-voice-execution-state", execution.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def invoke(app, scope, body, maximum=65 * 1024**2):
    consumed = False
    never = asyncio.Event()
    result = {"status": 500, "body": bytearray(), "content_type": "application/json"}

    async def receive():
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        await never.wait()

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
            result["content_type"] = (
                dict(message.get("headers", []))
                .get(b"content-type", b"application/octet-stream")
                .decode()
            )
        elif message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            if len(result["body"]) + len(chunk) > maximum:
                raise Rejected(502, "result_too_large", "finished")
            result["body"].extend(chunk)

    await app(scope, receive, send)
    return result["status"], result["content_type"], bytes(result["body"])


class ReceiverMiddleware:
    def __init__(self, app, config, state, validator):
        self.app, self.config, self.state, self.validator = (
            app,
            config,
            state,
            validator,
        )
        self.worker_busy = False
        self.admitted = {"high": 0, "normal": 0}

    def scope(self, content_type, query):
        route = "/transcribe" if self.config.operation == "stt" else "/synthesize"
        return {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": route,
            "raw_path": route.encode(),
            "query_string": query,
            "headers": [(b"content-type", content_type.encode())],
            "server": ("model", 8000),
            "client": ("receiver", 0),
            "root_path": "",
        }

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        engine = self.state.get("engine")
        if scope["path"] == "/health/ready":
            ready = self.config.mode == "worker" or (
                engine is not None and engine.ready and not engine.closing
            )
            return await reply(
                send, 200 if ready else 503, json.dumps({"ready": ready}).encode()
            )
        if scope["path"] == "/metrics" and engine is not None:
            status, media, body = await invoke(self.app, scope, b"")
            return await reply(
                send, status, body + b"\n" + generate_latest(engine.registry), media
            )
        path = "/transcribe" if self.config.operation == "stt" else "/synthesize"
        if scope["path"] != path or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        job_id = None
        admitted = None
        outcome = "internal_error"
        pending = []
        try:
            headers = {}
            for name, value in scope["headers"]:
                name = name.lower()
                if (
                    name in (b"authorization", b"idempotency-key", b"content-type")
                    and name in headers
                ):
                    raise Rejected(422, "duplicate_header")
                headers[name] = value.decode("latin-1")
            client = self.config.authenticate(headers.get(b"authorization", ""))
            if client is None:
                raise Rejected(401, "unauthorized")
            priority = client[1]
            slots = sum(i["slots"] for i in self.config.instances) or 1
            limit = (
                self.config.max_high if priority == "high" else self.config.max_normal
            ) + slots
            if self.admitted[priority] >= limit:
                raise Rejected(429, "request_limit")
            self.admitted[priority] += 1
            admitted = priority
            if self.config.mode == "queued" and (engine is None or not engine.ready):
                raise Rejected(503, "queue_unavailable")
            body = bytearray()
            upload_deadline = (
                asyncio.get_running_loop().time() + self.config.queue_timeout
            )
            while True:
                message = await asyncio.wait_for(
                    receive(),
                    timeout=max(0, upload_deadline - asyncio.get_running_loop().time()),
                )
                if message["type"] == "http.disconnect":
                    outcome = "cancelled"
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.config.max_request_bytes:
                    raise Rejected(413, "request_too_large")
                if not message.get("more_body", False):
                    break
            body = bytes(body)
            message = None
            content_type = headers.get(b"content-type", "")
            canonical = await self.validate(scope, body, content_type)
            if self.config.mode == "worker":
                # A worker is private and never grows another waiting queue.
                if self.worker_busy:
                    raise Rejected(429, "worker_busy")
                self.worker_busy = True
                task = asyncio.create_task(
                    invoke(
                        self.app,
                        self.scope(content_type, scope["query_string"]),
                        body,
                        self.config.max_result_bytes,
                    )
                )
                task.add_done_callback(self.worker_finished)
                try:
                    status, media, result = await asyncio.wait_for(
                        asyncio.shield(task), self.config.execution_timeout
                    )
                    outcome = "success" if status < 300 else "model_error"
                    return await reply(send, status, result, media, "finished")
                except TimeoutError:
                    # Keep the model reserved until the inference thread actually exits.
                    raise Rejected(504, "execution_timeout", "uncertain") from None
                finally:
                    if task.done():
                        self.worker_busy = False
            fingerprint = hashlib.sha256(
                scope["query_string"] + b"\0" + canonical
            ).hexdigest()
            abandoned = asyncio.Event()
            disconnect_task = asyncio.create_task(self.disconnected(receive, abandoned))
            pending = [disconnect_task]
            admission = asyncio.create_task(
                engine.submit(
                    client[0],
                    client[1],
                    headers.get(b"idempotency-key", ""),
                    body,
                    content_type,
                    scope["query_string"],
                    fingerprint=fingerprint,
                    abandoned=abandoned,
                )
            )
            try:
                job_id = await asyncio.shield(admission)
            except asyncio.CancelledError:
                # Complete critical persistence without starting a lost owner's job.
                abandoned.set()
                while not admission.done():
                    try:
                        await asyncio.shield(admission)
                    except asyncio.CancelledError:
                        continue
                job_id = admission.result()
                raise
            # Waiting HTTP owners retain only the job ID, not the uploaded audio.
            body = canonical = None
            if abandoned.is_set():
                outcome = "cancelled"
                return
            result_task = asyncio.create_task(engine.response(job_id))
            pending.append(result_task)
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if disconnect_task in done:
                outcome = "cancelled"
                return
            status, media, result, execution = result_task.result()
            outcome = "success" if status < 300 else "model_error"
            await reply(send, status, result, media, execution)
        except Rejected as error:
            outcome = error.reason
            await reply(
                send,
                error.status,
                json.dumps({"error": error.reason}).encode(),
                execution=error.execution,
            )
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except TimeoutError:
            outcome = "upload_timeout"
            await reply(send, 408, b'{"error":"upload_timeout"}')
        finally:
            if engine is not None:
                engine.requests.labels(self.config.operation, outcome).inc()
            if admitted is not None:
                self.admitted[admitted] -= 1
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if job_id is not None:
                await asyncio.shield(engine.detach(job_id))

    def worker_finished(self, task):
        self.worker_busy = False
        if not task.cancelled():
            task.exception()

    async def validate(self, scope, body, content_type):
        try:
            query = parse_qs(scope["query_string"].decode(), keep_blank_values=True)
        except (ValueError, UnicodeError):
            raise Rejected(422, "invalid_query") from None
        allowed = (
            {"model", "word_timestamps"} if self.config.operation == "stt" else set()
        )
        if set(query) - allowed or any(len(values) != 1 for values in query.values()):
            raise Rejected(422, "invalid_query")
        if "word_timestamps" in query and query["word_timestamps"][0].lower() not in {
            "true",
            "false",
            "1",
            "0",
            "yes",
            "no",
            "on",
            "off",
        }:
            raise Rejected(422, "invalid_query")
        if self.config.operation == "tts":
            if content_type.split(";", 1)[0].strip() != "application/json":
                raise Rejected(415, "unsupported_media_type")
            try:
                data = json.loads(body)
                self.validator(data)
                return json.dumps(
                    data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            except (ValueError, TypeError):
                raise Rejected(422, "invalid_synthesis_request") from None
        if not content_type.startswith("multipart/form-data;"):
            raise Rejected(415, "unsupported_media_type")
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        try:
            async with Request(scope, receive).form(
                max_files=1, max_fields=8, max_part_size=self.config.max_request_bytes
            ) as form:
                file = form.get("file")
                if not hasattr(file, "read"):
                    raise ValueError()
                audio = await file.read(self.config.max_request_bytes + 1)
                if not audio or len(audio) > 25 * 1024**2:
                    raise ValueError()
                model = query.get("model", [None])[0]
                if model is not None:
                    self.validator(model)
                return (
                    (file.content_type or "application/octet-stream").encode()
                    + b"\0"
                    + audio
                )
        except Exception:
            raise Rejected(422, "invalid_transcription_request") from None

    @staticmethod
    async def disconnected(receive, abandoned):
        while True:
            if (await receive())["type"] == "http.disconnect":
                abandoned.set()
                return


def install(app, config, validator, journal_factory=KafkaJournal):
    state = {}
    original = app.router.lifespan_context
    # Middleware is added last so admission wraps the existing metrics/model app.
    app.add_middleware(
        ReceiverMiddleware, config=config, state=state, validator=validator
    )

    @asynccontextmanager
    async def lifespan(application):
        if config.mode == "worker":
            async with original(application):
                yield
            return

        async def execute(body, content_type, query):
            # Inner ASGI app excludes admission, avoiding recursive enqueue.
            middleware = application.middleware_stack
            while not isinstance(middleware, ReceiverMiddleware):
                middleware = middleware.app
            return await invoke(
                middleware.app,
                middleware.scope(content_type, query),
                body,
                config.max_result_bytes,
            )

        # Acquire the volume before initializing Kafka ownership or loading GPU
        # weights. A duplicate process must fail before consuming model memory.
        engine = await asyncio.to_thread(
            Engine,
            config,
            lambda: journal_factory(config.brokers, config.topic),
            execute,
        )
        state["engine"] = engine
        application.state.receiver = engine
        try:
            async with original(application):
                await engine.start()
                try:
                    yield
                finally:
                    await engine.close()
        finally:
            # Also release ownership if loading the host model failed.
            await engine.close()

    app.router.lifespan_context = lifespan
    return app
