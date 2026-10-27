"""Read-only host checks before pulling a large image or downloading weights."""

import json
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSIONS = json.loads((ROOT / "versions.json").read_text())


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def version_tuple(value):
    return tuple(int(x) for x in value.strip().split("."))


def check():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Нужен Linux x86_64; этот набор не рассчитан на macOS/Windows/ARM.")
    for name in ("docker", "nvidia-smi"):
        if shutil.which(name) is None:
            raise RuntimeError(f"Не найдена команда {name}. Сначала установите её по README.")
    command("docker", "info", "--format", "{{.ServerVersion}}")
    command("docker", "compose", "version")
    if "--wait-timeout" not in command("docker", "compose", "up", "--help"):
        raise RuntimeError("Обновите плагин Docker Compose: нужен up --wait --wait-timeout.")
    compose = json.loads(command("docker", "compose", "-f", str(ROOT / "compose.yaml"), "config", "--format", "json"))
    service = compose["services"]["omni"]
    gpu_id = service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"][0]
    gpu = command("nvidia-smi", "-i", str(gpu_id),
                  "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits")
    if len(gpu.splitlines()) != 1:
        raise RuntimeError("GPU_ID должен выбирать ровно одну GPU.")
    name, total, free, driver = [s.strip() for s in gpu.split(",")]
    if not any(card in name for card in ("A100", "H100")) or float(total) < 76 * 1024:
        raise RuntimeError(f"Нужна A100/H100 80 ГБ без MIG. Обнаружена {name}, {total} MiB.")
    if version_tuple(driver) < version_tuple(VERSIONS["minimum_driver"]):
        raise RuntimeError(f"Драйвер {driver} слишком старый. Для выбранного CUDA 13 стека нужен {VERSIONS['minimum_driver']} или новее.")
    if float(free) < float(total) * 0.90:
        raise RuntimeError("GPU занята: свободно меньше 90% памяти. Остановите использующие её сервисы.")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    ram_gib = int(meminfo["MemTotal"].split()[0]) / 1024**2
    if ram_gib < 110:
        raise RuntimeError(f"Для этого профиля требуется сервер с RAM 128 ГБ. Обнаружено {ram_gib:.1f} GiB.")
    model_volume = next(v for v in service["volumes"] if v["target"] == "/models")
    data_dir = Path(model_volume["source"])
    data_dir.mkdir(parents=True, exist_ok=True)
    data_free = shutil.disk_usage(data_dir).free / 1024**3
    docker_dir = Path(command("docker", "info", "--format", "{{.DockerRootDir}}"))
    try:
        docker_free = shutil.disk_usage(docker_dir).free / 1024**3
        same_disk = docker_dir.stat().st_dev == data_dir.stat().st_dev
    except PermissionError as error:
        raise RuntimeError("Нет доступа для проверки диска Docker. Выполните проверку с sudo или предоставьте доступ к каталогу Docker.") from error
    model_ready = (data_dir / "prepared.json").is_file()
    required_data = 5 if model_ready else 40
    image_present = subprocess.run(
        ["docker", "image", "inspect", service["image"]],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    required_docker = 5 if image_present else 40
    if same_disk:
        if data_free < required_data + required_docker:
            raise RuntimeError(f"На общем диске весов и Docker нужно {required_data + required_docker} GiB свободно; есть {data_free:.1f}.")
    elif data_free < required_data or docker_free < required_docker:
        raise RuntimeError(f"Нужно {required_data} GiB на диске весов и {required_docker} GiB на диске Docker.")
    port = int(service["ports"][0]["published"])
    if port == int(compose["services"]["api"]["ports"][0]["published"]):
        raise RuntimeError("S2S_PORT и S2S_API_PORT должны быть разными.")
    for port_config in (service["ports"][0], compose["services"]["api"]["ports"][0]):
        checked_port = int(port_config["published"])
        with socket.socket() as sock:
            try:
                sock.bind((port_config.get("host_ip", "0.0.0.0"), checked_port))
            except OSError as error:
                raise RuntimeError(f"Порт {checked_port} занят. Если Omni уже запущен, используйте ./run.sh verify; иначе измените порт в .env.") from error
    report = {"gpu": name, "gpu_id": gpu_id, "driver": driver,
              "ram_gib": round(ram_gib, 1), "data_dir": str(data_dir),
              "data_free_gib": round(data_free, 1), "docker_free_gib": round(docker_free, 1), "port": port}
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "host-check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Проверка хоста пройдена. Доступ GPU из контейнера проверяется после сборки образа.")


if __name__ == "__main__":
    try:
        check()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"ОШИБКА: {error}") from error
