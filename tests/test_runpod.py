"""RunPod orchestration (spec 5): launch, poll, teardown verification.

No live RunPod calls, ever - every test substitutes a fake transport and
asserts on what was requested. This is the "no network access" principle the
rest of the suite already follows, applied to the one new external service.
"""
from __future__ import annotations

import pytest

from solbot.runpod import RunPodClient, RunPodError


class FakeTransport:
    """Records every request and answers from a small in-memory RunPod model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.volumes: dict[str, dict] = {}
        self.pods: dict[str, dict] = {}
        self._next_id = 1
        # Test knobs: pods finish (stop being RUNNING) after this many status polls.
        self.pod_polls_until_stopped: dict[str, int] = {}
        self._poll_counts: dict[str, int] = {}

    def _id(self, prefix: str) -> str:
        value = f"{prefix}-{self._next_id}"
        self._next_id += 1
        return value

    def request(self, method: str, path: str, *, json: dict | None = None) -> dict:
        self.calls.append((method, path, json or {}))

        if method == "POST" and path == "/networkvolumes":
            vol_id = self._id("vol")
            self.volumes[vol_id] = {"id": vol_id, "dataCenterId": json.get("dataCenterId", "US-CA-1")}
            return dict(self.volumes[vol_id])

        if method == "DELETE" and path.startswith("/networkvolumes/"):
            vol_id = path.rsplit("/", 1)[-1]
            if vol_id not in self.volumes:
                raise RunPodError("volume not found")
            del self.volumes[vol_id]
            return {}

        if method == "GET" and path.startswith("/networkvolumes/"):
            vol_id = path.rsplit("/", 1)[-1]
            if vol_id not in self.volumes:
                raise RunPodError("volume not found")
            return dict(self.volumes[vol_id])

        if method == "POST" and path == "/pods":
            pod_id = self._id("pod")
            self.pods[pod_id] = {"id": pod_id, "status": "RUNNING"}
            self._poll_counts[pod_id] = 0
            return dict(self.pods[pod_id])

        if method == "GET" and path.startswith("/pods/"):
            pod_id = path.rsplit("/", 1)[-1]
            if pod_id not in self.pods:
                raise RunPodError("pod not found")
            self._poll_counts[pod_id] = self._poll_counts.get(pod_id, 0) + 1
            stop_after = self.pod_polls_until_stopped.get(pod_id)
            if stop_after is not None and self._poll_counts[pod_id] >= stop_after:
                self.pods[pod_id]["status"] = "EXITED"
            return dict(self.pods[pod_id])

        if method == "POST" and path.endswith("/stop"):
            pod_id = path.split("/")[2]
            if pod_id in self.pods:
                self.pods[pod_id]["status"] = "EXITED"
            return {}

        if method == "DELETE" and path.startswith("/pods/"):
            pod_id = path.rsplit("/", 1)[-1]
            self.pods.pop(pod_id, None)
            return {}

        raise RunPodError(f"unhandled fake request: {method} {path}")


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def client(transport) -> RunPodClient:
    return RunPodClient(
        api_key="test-key", transport=transport,
        poll_interval_seconds=0.01, max_poll_seconds=0.05,
    )


# --------------------------------------------------------------------------
def test_create_and_delete_network_volume(client, transport):
    volume = client.create_network_volume("wfmc-batch-0", size_gb=20, data_center_id="US-CA-1")
    assert volume["id"] in transport.volumes

    client.delete_network_volume(volume["id"])
    assert volume["id"] not in transport.volumes
    assert not client.volume_exists(volume["id"])


def test_create_pod_and_check_status(client, transport):
    pod = client.create_pod(
        name="wfmc-batch-0", gpu_type="NVIDIA RTX 4090", volume_id="vol-1", env={"A": "B"}
    )
    assert client.pod_status(pod["id"]) == "RUNNING"
    assert client.pod_exists_and_running(pod["id"])


def test_terminate_pod_stops_and_removes_it(client, transport):
    pod = client.create_pod(name="x", gpu_type="g", volume_id="v", env={})
    client.terminate_pod(pod["id"])
    assert not client.pod_exists_and_running(pod["id"])


# --------------------------------------------------------------------------
# Teardown verification - the spec's explicit "confirm no worker or volume is
# left running" step.
# --------------------------------------------------------------------------
def test_verify_teardown_reports_clean_when_workers_self_terminate(client, transport):
    volume = client.create_network_volume("v0", size_gb=10, data_center_id="US-CA-1")
    pod = client.create_pod(name="p0", gpu_type="g", volume_id=volume["id"], env={})
    # The worker's own entrypoint script stops itself after its first poll.
    transport.pod_polls_until_stopped[pod["id"]] = 1

    result = client.verify_teardown([{"pod_id": pod["id"], "volume_id": volume["id"]}])

    assert result["clean"] is True
    assert not client.pod_exists_and_running(pod["id"])
    assert not client.volume_exists(volume["id"])


def test_verify_teardown_force_terminates_a_worker_that_did_not_self_stop(client, transport):
    volume = client.create_network_volume("v1", size_gb=10, data_center_id="US-CA-1")
    pod = client.create_pod(name="p1", gpu_type="g", volume_id=volume["id"], env={})
    # Never self-terminates within the poll window.
    transport.pod_polls_until_stopped[pod["id"]] = None

    result = client.verify_teardown([{"pod_id": pod["id"], "volume_id": volume["id"]}])

    # wait_for_completion's safety net stops it even though it never self-exited.
    assert result["clean"] is True
    assert any(call[:2] == ("POST", f"/pods/{pod['id']}/stop") for call in transport.calls)


def test_verify_teardown_reports_not_clean_when_a_volume_survives(client, transport, monkeypatch):
    volume = client.create_network_volume("v2", size_gb=10, data_center_id="US-CA-1")
    pod = client.create_pod(name="p2", gpu_type="g", volume_id=volume["id"], env={})
    transport.pod_polls_until_stopped[pod["id"]] = 1

    # Simulate RunPod refusing the delete (e.g. a transient API error) so the
    # volume is still there afterward.
    def fail_delete(volume_id: str) -> None:
        raise RunPodError("simulated failure")

    monkeypatch.setattr(client, "delete_network_volume", fail_delete)
    result = client.verify_teardown([{"pod_id": pod["id"], "volume_id": volume["id"]}])

    assert result["clean"] is False
    assert volume["id"] in result["detail"]
