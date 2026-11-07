import asyncio
import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs

import httpx
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Rejected(Exception):
    def __init__(self, status, reason, execution="not-started"):
        self.status, self.reason, self.execution = status, reason, execution


class Blobs:
    def __init__(self, path, limit):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.path / ".lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise RuntimeError("receiver data volume already owned") from None
        self.limit = limit
        self.used = sum(p.stat().st_size for p in self.path.glob("*.blob"))
        for p in self.path.glob(".tmp-*"):
            p.unlink()

    def file(self, name):
        if (
            len(name) != 69
            or not name.endswith(".blob")
            or any(c not in "0123456789abcdef" for c in name[:-5])
        ):
            raise RuntimeError("invalid payload reference")
        return self.path / name

    def put(self, data):
        name = uuid.uuid4().hex + uuid.uuid4().hex + ".blob"
        if self.used + len(data) > self.limit:
            raise Rejected(503, "payload_quota")
        fd, temp = tempfile.mkstemp(prefix=".tmp-", dir=self.path)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, self.file(name))
            self.sync()
            self.used += len(data)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        return name

    def get(self, name, maximum):
        with self.file(name).open("rb") as file:
            data = file.read(maximum + 1)
        if len(data) > maximum:
            raise RuntimeError("stored payload exceeds limit")
        return data

    def remove(self, name):
        path = self.file(name)
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            self.used -= size
            self.sync()

    def sync(self):
        fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def close(self):
        self.lock.close()


class Engine:
    def __init__(self, config, journal, execute):
        self.config, self.execute = config, execute
        self.blobs = Blobs(config.data_dir, config.max_payload_bytes)
        try:
            self.journal = journal() if callable(journal) else journal
        except BaseException:
            self.blobs.close()
            raise
        self.jobs, self.waiters, self.tasks = {}, {}, set()
        self.lock, self.changed, self.fatal = (
            asyncio.Lock(),
            asyncio.Event(),
            asyncio.Event(),
        )
        self.ready = False
        self.closing = False
        self.closed = False
        self.instances = config.instances or [{"id": "local", "slots": 1}]
        self.registry = CollectorRegistry()
        self.depth = Gauge(
            "receiver_queue_depth",
            "Waiting jobs",
            ["operation", "priority"],
            registry=self.registry,
        )
        self.busy = Gauge(
            "receiver_instance_busy",
            "Reserved slots including uncertain outcomes",
            ["operation", "model_instance"],
            registry=self.registry,
        )
        self.outcomes = Counter(
            "receiver_jobs_total",
            "Accepted job outcomes",
            ["operation", "priority", "outcome"],
            registry=self.registry,
        )
        self.requests = Counter(
            "receiver_requests_total",
            "HTTP admission outcomes",
            ["operation", "outcome"],
            registry=self.registry,
        )
        self.wait_time = Histogram(
            "receiver_queue_wait_seconds",
            "Queue wait",
            ["operation", "priority"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "receiver_execution_seconds",
            "Model execution",
            ["operation", "priority"],
            registry=self.registry,
        )
        self.payload = Gauge(
            "receiver_payload_bytes", "Payload bytes", registry=self.registry
        )
        self.available = Gauge(
            "receiver_ready", "Durable queue ready", registry=self.registry
        )
        self.dispatcher = None
        self.http = httpx.AsyncClient(
            follow_redirects=False, timeout=config.execution_timeout
        )

    def fail(self):
        self.ready = False
        self.available.set(0)
        self.fatal.set()

    def store_payload(self, data):
        try:
            return self.blobs.put(data)
        except Rejected:
            raise
        except Exception:
            # A rename/fsync failure leaves filesystem usage uncertain.
            # Stop accepting tasks until a restart has scanned the volume.
            self.fail()
            raise Rejected(503, "payload_write_failed") from None

    async def persist(self, job, delete=False):
        try:
            await asyncio.to_thread(self.journal.append, dict(job), delete)
        except Exception:
            self.ready = False
            self.available.set(0)
            self.fatal.set()
            raise Rejected(503, "durable_write_unconfirmed", "uncertain") from None

    async def start(self, dispatch=True):
        try:
            self.jobs = await asyncio.to_thread(self.journal.replay)
            configured = {x["id"]: x for x in self.instances}
            for key, job in self.jobs.items():
                if (
                    key != job["id"]
                    or job.get("operation") != self.config.operation
                    or job["priority"] not in ("high", "normal")
                    or job["state"] not in ("queued", "running", "succeeded", "failed")
                ):
                    raise RuntimeError("invalid receiver state")
                if job["state"] in ("queued", "running"):
                    # HTTP owners were lost. Never replay an uncertain generation.
                    remote = job["state"] == "running"
                    job.update(
                        state="failed",
                        status=503,
                        reason="interrupted",
                        held=remote,
                        finished=time.time(),
                        execution="uncertain" if remote else "not-started",
                    )
                    await self.persist(job)
                if job.get("output"):
                    if (
                        self.blobs.file(job["output"]).stat().st_size
                        > self.config.max_result_bytes
                    ):
                        raise RuntimeError("stored result exceeds configured limit")
                if job.get("held") and job["instance"] not in configured:
                    raise RuntimeError("held instance missing from configuration")
            for instance in self.instances:
                held = sum(
                    j.get("held", False) and j.get("instance") == instance["id"]
                    for j in self.jobs.values()
                )
                if held > instance["slots"]:
                    raise RuntimeError("configured slots are below held reservations")
            referenced = {
                j[k]
                for j in self.jobs.values()
                for k in ("input", "output")
                if j.get(k)
            }
            for path in self.blobs.path.glob("*.blob"):
                if path.name not in referenced:
                    self.blobs.remove(path.name)
            self.ready = True
            self.available.set(1)
            self.refresh()
            if dispatch:
                self.dispatcher = asyncio.create_task(self.dispatch())
        except BaseException:
            await self.close()
            raise

    def refresh(self):
        op = self.config.operation
        for priority in ("high", "normal"):
            self.depth.labels(op, priority).set(
                sum(
                    j["state"] == "queued" and j["priority"] == priority
                    for j in self.jobs.values()
                )
            )
        for instance in self.instances:
            self.busy.labels(op, instance["id"]).set(
                sum(
                    (j["state"] == "running" or j.get("held", False))
                    and j.get("instance") == instance["id"]
                    for j in self.jobs.values()
                )
            )
        self.payload.set(self.blobs.used)

    async def submit(
        self,
        client,
        priority,
        key,
        body,
        content_type,
        query,
        fingerprint=None,
        abandoned=None,
    ):
        fingerprint = (
            fingerprint
            or hashlib.sha256(
                content_type.encode() + b"\0" + query + b"\0" + body
            ).hexdigest()
        )
        key = key or uuid.uuid4().hex
        if (
            len(key) > 128
            or not key.isascii()
            or any(ord(c) < 33 or ord(c) > 126 for c in key)
        ):
            raise Rejected(422, "invalid_idempotency_key")
        job_id = hashlib.sha256((client + "\0" + key).encode()).hexdigest()
        async with self.lock:
            if not self.ready or self.closing:
                raise Rejected(503, "queue_unavailable")
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job["fingerprint"] != fingerprint:
                    raise Rejected(409, "idempotency_conflict")
            else:
                limit = (
                    self.config.max_high
                    if priority == "high"
                    else self.config.max_normal
                )
                if (
                    sum(
                        j["state"] == "queued" and j["priority"] == priority
                        for j in self.jobs.values()
                    )
                    >= limit
                ):
                    raise Rejected(429, "queue_full")
                selected = parse_qs(query.decode()).get("model", [None])[0]
                if not any(
                    not i.get("models") or selected is None or selected in i["models"]
                    for i in self.instances
                ):
                    raise Rejected(422, "no_compatible_model")
                payload = self.store_payload(body)
                job = dict(
                    id=job_id,
                    operation=self.config.operation,
                    fingerprint=fingerprint,
                    priority=priority,
                    state="queued",
                    input=payload,
                    content_type=content_type,
                    query=query.decode(),
                    model=selected,
                    created=time.time(),
                    held=False,
                )
                try:
                    await self.persist(job)
                except BaseException:
                    # Leave the file: an unconfirmed append may have reached Kafka.
                    raise
                self.jobs[job_id] = job
            self.waiters[job_id] = self.waiters.get(job_id, 0) + 1
            if (
                abandoned is not None
                and abandoned.is_set()
                and self.waiters[job_id] == 1
                and job["state"] == "queued"
            ):
                await self.finish(job, 503, "cancelled", "not-started")
            self.changed.set()
            self.refresh()
        return job_id

    async def response(self, job_id):
        while True:
            async with self.lock:
                job = self.jobs.get(job_id)
                if job is None:
                    raise Rejected(404, "result_expired")
                if not self.ready and job["state"] not in ("succeeded", "failed"):
                    raise Rejected(
                        503,
                        "queue_unavailable",
                        "uncertain" if job["state"] == "running" else "not-started",
                    )
                if job["state"] in ("succeeded", "failed"):
                    if job.get("output"):
                        return (
                            job["status"],
                            job["response_type"],
                            self.blobs.get(job["output"], self.config.max_result_bytes),
                            job.get("execution", "finished"),
                        )
                    raise Rejected(
                        job.get("status", 503),
                        job.get("reason", "execution_failed"),
                        job.get("execution", "finished"),
                    )
                deadline = job.get("started", job["created"]) + (
                    self.config.execution_timeout
                    if job["state"] == "running"
                    else self.config.queue_timeout
                )
                if time.time() >= deadline:
                    if job["state"] == "queued":
                        await self.finish(job, 503, "queue_timeout", "not-started")
                        self.changed.set()
                    else:
                        raise Rejected(504, "execution_timeout", "uncertain")
            await asyncio.sleep(0.025)

    async def detach(self, job_id):
        async with self.lock:
            self.waiters[job_id] = max(0, self.waiters.get(job_id, 1) - 1)
            job = self.jobs.get(job_id)
            if (
                not self.waiters[job_id]
                and job
                and job["state"] == "queued"
                and self.ready
            ):
                await self.finish(job, 503, "cancelled", "not-started")
                self.changed.set()

    async def finish(self, job, status, reason, execution, held=False):
        job.update(
            state="failed",
            status=status,
            reason=reason,
            execution=execution,
            held=held,
            finished=time.time(),
        )
        await self.persist(job)
        if job.get("input"):
            self.blobs.remove(job["input"])
        self.outcomes.labels(self.config.operation, job["priority"], reason).inc()
        self.refresh()

    async def dispatch(self):
        while not self.closing and self.ready:
            self.changed.clear()
            try:
                async with self.lock:
                    now = time.time()
                    for job in list(self.jobs.values()):
                        if (
                            job["state"] == "queued"
                            and now >= job["created"] + self.config.queue_timeout
                        ):
                            await self.finish(job, 503, "queue_timeout", "not-started")
                        elif (
                            job["state"] in ("failed", "succeeded")
                            and not job.get("held")
                            and not self.waiters.get(job["id"])
                            and now - job["finished"] >= self.config.retention
                        ):
                            await self.persist(job, True)
                            for field in ("input", "output"):
                                if job.get(field):
                                    self.blobs.remove(job[field])
                            self.jobs.pop(job["id"])
                            self.waiters.pop(job["id"], None)
                    pending = sorted(
                        (j for j in self.jobs.values() if j["state"] == "queued"),
                        key=lambda j: (j["priority"] != "high", j["created"]),
                    )
                    for instance in self.instances:
                        used = sum(
                            (j["state"] == "running" or j.get("held", False))
                            and j.get("instance") == instance["id"]
                            for j in self.jobs.values()
                        )
                        for job in list(pending):
                            if used >= instance["slots"]:
                                break
                            if (
                                instance.get("models")
                                and job["model"]
                                and job["model"] not in instance["models"]
                            ):
                                continue
                            job.update(
                                state="running",
                                instance=instance["id"],
                                started=time.time(),
                            )
                            await self.persist(job)
                            self.wait_time.labels(
                                self.config.operation, job["priority"]
                            ).observe(job["started"] - job["created"])
                            task = asyncio.create_task(self.run(job, instance))
                            self.tasks.add(task)
                            task.add_done_callback(self.tasks.discard)
                            pending.remove(job)
                            used += 1
                    self.refresh()
            except Exception:
                self.ready = False
                self.available.set(0)
                self.fatal.set()
                logging.error(
                    json.dumps(
                        {
                            "event": "receiver_unavailable",
                            "reason": "state_transition_failed",
                        }
                    )
                )
                return
            try:
                await asyncio.wait_for(self.changed.wait(), timeout=0.2)
            except TimeoutError:
                pass

    async def run(self, job, instance):
        try:
            body = self.blobs.get(job["input"], self.config.max_request_bytes)
            if instance["id"] == "local":
                result = await self.execute(
                    body, job["content_type"], job["query"].encode()
                )
            else:
                # Remote worker has its own private token and no additional queue.
                async with self.http.stream(
                    "POST",
                    instance["url"],
                    params=job["query"],
                    content=body,
                    headers={
                        "Authorization": "Bearer " + instance["token"],
                        "Content-Type": job["content_type"],
                    },
                ) as response:
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > self.config.max_result_bytes:
                            raise RuntimeError("result too large")
                    if response.headers.get("x-voice-execution-state") not in (
                        "finished",
                        "not-started",
                    ):
                        raise Rejected(503, "remote_uncertain", "uncertain")
                    result = (
                        response.status_code,
                        response.headers.get(
                            "content-type", "application/octet-stream"
                        ),
                        bytes(data),
                    )
            status, content_type, data = result
            if len(data) > self.config.max_result_bytes:
                raise Rejected(502, "result_too_large", "finished")
            async with self.lock:
                output = self.store_payload(data)
                completed = dict(
                    job,
                    state="succeeded" if 200 <= status < 300 else "failed",
                    output=output,
                    status=status,
                    response_type=content_type,
                    execution="finished",
                    finished=time.time(),
                )
                await self.persist(completed)
                job.update(completed)
                self.blobs.remove(job["input"])
                self.outcomes.labels(
                    self.config.operation,
                    job["priority"],
                    "success" if status < 300 else "model_error",
                ).inc()
        except Exception as error:
            async with self.lock:
                if job["state"] in ("succeeded", "failed"):
                    # Cleanup must not overwrite a committed result.
                    self.fail()
                elif self.ready:
                    remote = instance["id"] != "local" and not (
                        isinstance(error, Rejected) and error.execution == "finished"
                    )
                    try:
                        await self.finish(
                            job,
                            503,
                            "execution_failed",
                            "uncertain" if remote else "finished",
                            held=remote,
                        )
                    except Exception:
                        self.fatal.set()
        finally:
            self.duration.labels(self.config.operation, job["priority"]).observe(
                time.time() - job["started"]
            )
            self.refresh()
            self.changed.set()

    async def release_held(self, instance):
        async with self.lock:
            if instance not in {i["id"] for i in self.instances}:
                raise ValueError("unknown instance")
            for job in self.jobs.values():
                if job.get("held") and job["instance"] == instance:
                    job["held"] = False
                    await self.persist(job)
            self.refresh()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.closing = True
        self.ready = False
        self.available.set(0)
        self.changed.set()
        if self.dispatcher:
            await self.dispatcher
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        try:
            await self.http.aclose()
            await asyncio.to_thread(self.journal.close)
        finally:
            self.blobs.close()
