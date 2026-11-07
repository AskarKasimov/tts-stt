import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).parents[2]


def test_prometheus_targets_and_interval():
    text = (ROOT / "observability/prometheus.yml").read_text()
    assert "scrape_interval: 15s" in text
    assert 'targets: ["gigaam:8000"]' in text
    assert 'targets: ["voxcpm2:8000"]' in text


def test_dashboard_has_required_queries_and_disclaimer():
    dashboard = json.loads((ROOT / "observability/grafana/dashboards/speech-workload.json").read_text())
    raw = json.dumps(dashboard, ensure_ascii=False)
    for value in ("speech_requests_total", "speech_audio_seconds_total", "speech_request_duration_seconds", "up", "error %", "90 дней", "GPU-часы", "receiver_queue_depth", "receiver_queue_wait_seconds", "receiver_instance_busy", "receiver_ready"):
        assert value in raw
    assert dashboard["uid"] == "speech-workload"


def test_compose_has_documented_grafana_port_and_no_gpu_monitoring():
    env = os.environ.copy()
    env["GRAFANA_ADMIN_PASSWORD"] = "test-only"
    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yaml", "config", "--format", "json"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    compose = json.loads(result.stdout)
    services = compose["services"]
    assert services["grafana"]["ports"] == [{"mode": "ingress", "target": 3000, "published": "18473", "protocol": "tcp"}]
    assert "ports" not in services["prometheus"]
    assert services["prometheus"]["command"][-1] == "--storage.tsdb.retention.time=90d"
    for name in ("grafana", "prometheus"):
        assert all(key not in services[name] for key in ("gpus", "devices", "runtime"))
        assert "deploy" not in services[name]
