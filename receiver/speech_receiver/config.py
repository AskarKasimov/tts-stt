import hmac
import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def mapping(loader, node):
    result = {}
    for key, value in node.value:
        key = loader.construct_object(key)
        if not isinstance(key, str) or key in result:
            raise ValueError("configuration keys must be unique strings")
        result[key] = loader.construct_object(value)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


@dataclass
class Config:
    operation: str
    data_dir: str
    brokers: list[str]
    topic: str
    clients: dict
    instances: list
    mode: str = "queued"
    max_high: int = 32
    max_normal: int = 32
    max_payload_bytes: int = 2 * 1024**3
    max_request_bytes: int = 26 * 1024**2
    max_result_bytes: int = 65 * 1024**2
    queue_timeout: float = 120
    execution_timeout: float = 120
    retention: float = 86400
    host: str = "0.0.0.0"
    port: int = 8000

    def __post_init__(self):
        if self.operation not in ("stt", "tts") or self.mode not in (
            "queued",
            "worker",
        ):
            raise ValueError("invalid operation or mode")
        if not isinstance(self.clients, dict) or not self.clients:
            raise ValueError("at least one configured client is required")
        seen = set()
        for name, client in self.clients.items():
            if (
                not isinstance(name, str)
                or not isinstance(client, dict)
                or set(client) != {"token", "priority"}
            ):
                raise ValueError("invalid client definition")
            token = client["token"]
            if (
                not isinstance(token, str)
                or not 24 <= len(token) <= 4096
                or any(ord(c) < 33 or ord(c) > 126 for c in token)
                or token in seen
                or token.startswith("REPLACE")
            ):
                raise ValueError(
                    "client tokens must be unique ASCII secrets of at least 24 characters"
                )
            seen.add(token)
            if client["priority"] not in ("high", "normal"):
                raise ValueError("invalid client priority")
        if (
            not isinstance(self.brokers, list)
            or not self.brokers
            or not all(isinstance(x, str) and x for x in self.brokers)
            or not isinstance(self.topic, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,248}", self.topic)
            or not isinstance(self.data_dir, str)
            or not self.data_dir
            or not isinstance(self.host, str)
            or not self.host
        ):
            raise ValueError("brokers, topic and data_dir are required")
        for key in (
            "max_high",
            "max_normal",
            "max_payload_bytes",
            "max_request_bytes",
            "max_result_bytes",
            "port",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("queue_timeout", "execution_timeout", "retention"):
            value = getattr(self, key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be positive and finite")
        if self.port > 65535 or not isinstance(self.instances, list):
            raise ValueError("invalid port or instances list")
        ids = set()
        for instance in self.instances:
            if not isinstance(instance, dict):
                raise ValueError("instance must be a mapping")
            if (
                set(instance) - {"id", "url", "token", "slots", "models"}
                or not instance.get("id")
                or instance["id"] in ids
                or instance["id"] == "local"
            ):
                raise ValueError("instance IDs must be unique")
            ids.add(instance["id"])
            if (
                not instance.get("url", "").startswith(("http://", "https://"))
                or len(instance.get("token", "")) < 24
                or type(instance.get("slots")) is not int
                or instance["slots"] < 1
            ):
                raise ValueError(
                    "remote instances require URL, token and positive slots"
                )
            parsed = urlsplit(instance["url"])
            if (
                not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "instance URL must not contain credentials, query or fragment"
                )
            if (
                any(ord(c) < 33 or ord(c) > 126 for c in instance["token"])
                or instance["token"].startswith("REPLACE")
                or len(instance["token"]) > 4096
            ):
                raise ValueError("invalid worker token")
            if "models" in instance and (
                not isinstance(instance["models"], list)
                or not all(isinstance(x, str) for x in instance["models"])
            ):
                raise ValueError("instance models must be a list of strings")
        if self.mode == "worker" and self.instances:
            raise ValueError("worker cannot forward to other instances")

    def authenticate(self, header):
        token = header.removeprefix("Bearer ") if header.startswith("Bearer ") else ""
        for name, client in self.clients.items():
            if hmac.compare_digest(token.encode(), client["token"].encode()):
                return name, client["priority"]
        return None


def load(path):
    value = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return Config(**value)
