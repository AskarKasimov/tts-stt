"""Launcher used by both existing model images; no additional gateway service."""

import argparse
import asyncio
import importlib
import json
import logging

from .config import load
from .engine import Engine
from .http import install
from .journal import KafkaJournal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--release-held", metavar="INSTANCE")
    parser.add_argument("--upstream-stopped", action="store_true")
    args = parser.parse_args()
    config = load(args.config)
    if args.release_held:
        if not args.upstream_stopped or config.mode != "queued":
            parser.error("release requires queued mode and --upstream-stopped")

        async def release():
            async def unavailable(*_):
                raise RuntimeError("offline maintenance")

            engine = await asyncio.to_thread(
                Engine,
                config,
                lambda: KafkaJournal(config.brokers, config.topic),
                unavailable,
            )
            await engine.start(dispatch=False)
            try:
                await engine.release_held(args.release_held)
            finally:
                await engine.close()

        asyncio.run(release())
        return
    import uvicorn

    api = importlib.import_module("api")
    if config.operation == "stt":
        app = api.create_app(preload=not bool(config.instances))
        validator = api.resolve_model_name
    else:
        app = api.app
        validator = api.SynthesisRequest.model_validate
    app = install(app, config, validator)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.host,
            port=config.port,
            workers=1,
            access_log=False,
            log_config=None,
        )
    )

    async def serve():
        task = asyncio.create_task(server.serve())
        while not task.done():
            engine = getattr(app.state, "receiver", None)
            if engine is not None and engine.fatal.is_set():
                server.should_exit = True
            await asyncio.sleep(0.1)
        await task

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(
        json.dumps(
            {
                "event": "receiver_starting",
                "operation": config.operation,
                "mode": config.mode,
            }
        )
    )
    asyncio.run(serve())


if __name__ == "__main__":
    main()
