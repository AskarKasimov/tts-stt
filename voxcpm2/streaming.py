"""Bounded thread-to-ASGI bridge. The producer owns generator closure and model lock."""
import asyncio
import base64
import json
import queue
import threading
import time

import numpy as np

PCM_BLOCK_BYTES = 64 * 1024
MAX_PCM_BYTES = 64 * 1024 * 1024


class StreamCancelled(Exception):
    pass


def line(event):
    return (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")


def pcm_blocks(audio):
    values = np.asarray(audio)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise RuntimeError("Invalid streaming waveform")
    # Convert one bounded slice at a time; a model chunk can be much larger.
    for offset in range(0, values.size, PCM_BLOCK_BYTES // 2):
        block = np.clip(values[offset:offset + PCM_BLOCK_BYTES // 2], -1, 1)
        yield np.clip(np.rint(block * 32768), -32768, 32767).astype("<i2").tobytes()


class StreamJob:
    def __init__(self, producer, timeout=5.0):
        self.queue = queue.Queue(maxsize=8)
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.timeout = timeout
        self.outcome = "interrupted"
        self.samples = 0
        self.chunks = 0
        self.sample_rate = None
        self.started = time.perf_counter()
        self.first_audio = None
        self.thread = threading.Thread(target=self._run, args=(producer,), name="voxcpm-stream", daemon=True)
        self.thread.start()

    def _run(self, producer):
        try:
            producer(self)
        finally:
            self.done.set()

    def check(self):
        if self.cancel.is_set():
            raise StreamCancelled()

    def put(self, event):
        value = line(event)
        deadline = time.monotonic() + self.timeout
        while True:
            self.check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.cancel.set()
                raise StreamCancelled("Stream drain timed out")
            try:
                self.queue.put(value, timeout=min(0.05, remaining))
                return
            except queue.Full:
                continue

    def audio(self, waveform):
        for pcm in pcm_blocks(waveform):
            self.check()
            if (self.samples * 2) + len(pcm) > MAX_PCM_BYTES:
                raise RuntimeError("Stream exceeds decoded PCM limit")
            self.put({"type": "audio", "seq": self.chunks, "data": base64.b64encode(pcm).decode("ascii")})
            if self.first_audio is None:
                self.first_audio = time.perf_counter()
            self.chunks += 1
            self.samples += len(pcm) // 2

    async def body(self):
        try:
            while True:
                try:
                    value = self.queue.get_nowait()
                except queue.Empty:
                    if self.done.is_set():
                        break
                    await asyncio.sleep(0.01)
                    continue
                yield value
        finally:
            self.cancel.set()
