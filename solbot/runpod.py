"""RunPod orchestration for the monthly full parameter-space retest (spec 5).

The droplet is the only orchestrator: it creates a network volume per coin
batch, uploads that batch's Parquet chunk to it, launches a GPU pod that runs
``python -m solopt run`` against the mounted volume (reporting its feed and
finished bundle back to this droplet over HTTP, like any other reporting
client - see :mod:`solopt.report`), and afterward tears the pod and volume
down and *verifies* the teardown rather than assuming it happened.

Every network call goes through :attr:`RunPodClient.transport`, an injectable
seam so tests can substitute a fake and assert on what was requested without
touching RunPod's API - see the module docstring's spirit in the tests file:
no live RunPod calls in tests, ever.

**Before trusting this in production**, benchmark an actual run on RunPod and
confirm the GPU tier, the image reference, and the exact REST/GraphQL shapes
below against RunPod's current API docs - this module encodes the
orchestration *logic* precisely; the wire format is believed correct as of
this writing but RunPod's API has changed shape before and deserves a live
check the first time this runs for real.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)

API_BASE = "https://rest.runpod.io/v1"
POD_RUNNING_STATES = {"CREATED", "RESTARTING", "RUNNING", "PENDING"}
DEFAULT_IMAGE = "ghcr.io/example-org/solopt-worker:latest"
MAX_POLL_SECONDS = 4 * 3600
POLL_INTERVAL_SECONDS = 30


class RunPodError(RuntimeError):
    """RunPod refused a request or could not be reached."""


class Transport(Protocol):
    def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


@dataclass
class HttpxTransport:
    """The real transport: httpx against RunPod's REST API."""

    api_key: str
    timeout: float = 30.0

    def request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        import httpx

        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = httpx.request(method, url, json=json, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise RunPodError(f"could not reach RunPod: {exc}") from exc
        if response.status_code >= 400:
            raise RunPodError(f"RunPod {method} {path} returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            return response.json() if response.content else {}
        except ValueError as exc:
            raise RunPodError(f"RunPod {path} did not return JSON") from exc


@dataclass
class RunPodClient:
    api_key: str
    transport: Transport = field(default=None)  # type: ignore[assignment]
    image: str = DEFAULT_IMAGE
    # Overridable so tests can shrink these from hours/seconds to milliseconds
    # rather than actually sleeping through a poll loop.
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS
    max_poll_seconds: float = MAX_POLL_SECONDS

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = HttpxTransport(self.api_key)

    # ------------------------------------------------------------------
    # Network volumes
    # ------------------------------------------------------------------
    def create_network_volume(self, name: str, *, size_gb: int, data_center_id: str) -> dict[str, Any]:
        return self.transport.request(
            "POST", "/networkvolumes",
            json={"name": name, "size": int(size_gb), "dataCenterId": data_center_id},
        )

    def delete_network_volume(self, volume_id: str) -> None:
        self.transport.request("DELETE", f"/networkvolumes/{volume_id}")

    def volume_exists(self, volume_id: str) -> bool:
        try:
            self.transport.request("GET", f"/networkvolumes/{volume_id}")
            return True
        except RunPodError:
            return False

    def upload_batch_data(
        self, volume: dict[str, Any], local_dir: Path, *, s3_access_key: str, s3_secret_key: str
    ) -> None:
        """Push one batch's materialized Parquet directory onto its volume.

        RunPod exposes each network volume as an S3-compatible bucket in its
        datacenter; ``boto3`` is the standard client for that, same as
        RunPod's own docs recommend, and is only imported here - the daily,
        no-GPU path never needs it installed.
        """
        try:
            import boto3
        except ImportError as exc:
            raise RunPodError(
                "boto3 is required to upload data to a RunPod network volume "
                "(pip install boto3)"
            ) from exc

        data_center = volume.get("dataCenterId", "").lower()
        endpoint = f"https://s3api-{data_center}.runpod.io"
        s3 = boto3.client(
            "s3", endpoint_url=endpoint,
            aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key,
        )
        bucket = volume["id"]
        for path in Path(local_dir).iterdir():
            if path.is_file():
                s3.upload_file(str(path), bucket, path.name)

    # ------------------------------------------------------------------
    # Pods
    # ------------------------------------------------------------------
    def create_pod(
        self,
        *,
        name: str,
        gpu_type: str,
        volume_id: str,
        env: dict[str, str],
    ) -> dict[str, Any]:
        return self.transport.request(
            "POST", "/pods",
            json={
                "name": name,
                "imageName": self.image,
                "gpuTypeIds": [gpu_type],
                "networkVolumeId": volume_id,
                "env": [{"key": k, "value": v} for k, v in env.items()],
                "containerDiskInGb": 20,
                "volumeMountPath": "/data",
            },
        )

    def pod_status(self, pod_id: str) -> str:
        data = self.transport.request("GET", f"/pods/{pod_id}")
        return str(data.get("desiredStatus") or data.get("status") or "UNKNOWN").upper()

    def terminate_pod(self, pod_id: str) -> None:
        self.transport.request("POST", f"/pods/{pod_id}/stop")
        self.transport.request("DELETE", f"/pods/{pod_id}")

    def pod_exists_and_running(self, pod_id: str) -> bool:
        try:
            return self.pod_status(pod_id) in POD_RUNNING_STATES
        except RunPodError:
            return False

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def launch_batch(
        self,
        *,
        batch_index: int,
        mints: list[str],
        candles: Any,
        interval: str,
        gpu_type: str,
        report_run_id: int,
        droplet_url: str = "",
        droplet_token: str = "",
        data_center_id: str = "US-CA-1",
        s3_access_key: str = "",
        s3_secret_key: str = "",
    ) -> dict[str, Any]:
        """Ship one coin batch's data up and start its worker.

        Returns ``{"pod_id", "volume_id", "mints"}`` so the caller can poll
        and, later, verify teardown.
        """
        import tempfile

        volume = self.create_network_volume(
            f"wfmc-batch-{batch_index}-{int(time.time())}", size_gb=20,
            data_center_id=data_center_id,
        )
        volume_id = volume["id"]

        with tempfile.TemporaryDirectory(prefix="wfmc-batch-") as tmp:
            candles.materialize_bundle(mints, interval, tmp)
            if s3_access_key and s3_secret_key:
                self.upload_batch_data(
                    volume, Path(tmp), s3_access_key=s3_access_key, s3_secret_key=s3_secret_key
                )
            else:
                log.warning(
                    "no RunPod S3 credentials configured; batch %d's data was not "
                    "uploaded (dry-run/test mode)", batch_index,
                )

        pod = self.create_pod(
            name=f"wfmc-batch-{batch_index}",
            gpu_type=gpu_type,
            volume_id=volume_id,
            env={
                "SOLOPT_DROPLET_URL": droplet_url,
                "SOLOPT_TOKEN": droplet_token,
                "SOLOPT_REPORT_RUN_ID": str(report_run_id),
                "SOLOPT_CONFIG": "/data/solopt.json",
            },
        )
        return {"pod_id": pod["id"], "volume_id": volume_id, "mints": mints}

    def wait_for_completion(
        self, jobs: list[dict[str, Any]], *, max_seconds: float | None = None
    ) -> None:
        """Poll until every job's pod has stopped running, or the deadline passes.

        A worker is expected to terminate itself when its own run finishes;
        this is the safety net for one that does not, not the primary
        teardown mechanism.
        """
        deadline = time.monotonic() + (self.max_poll_seconds if max_seconds is None else max_seconds)
        pending = {j["pod_id"] for j in jobs}
        while pending and time.monotonic() < deadline:
            pending = {p for p in pending if self.pod_exists_and_running(p)}
            if pending:
                time.sleep(self.poll_interval_seconds)
        for pod_id in pending:
            log.warning("pod %s did not self-terminate; stopping it now", pod_id)
            try:
                self.terminate_pod(pod_id)
            except RunPodError as exc:
                log.warning("could not terminate pod %s: %s", pod_id, exc)

    def verify_teardown(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        """Confirm no worker or volume from ``jobs`` is left running.

        Logged either way (spec 5) - a clean result is itself worth recording,
        not just a failure.
        """
        self.wait_for_completion(jobs)
        still_running = [j["pod_id"] for j in jobs if self.pod_exists_and_running(j["pod_id"])]
        still_volumed = []
        for job in jobs:
            volume_id = job.get("volume_id")
            if not volume_id:
                continue
            try:
                self.delete_network_volume(volume_id)
            except RunPodError as exc:
                log.warning("could not delete volume %s: %s", volume_id, exc)
                if self.volume_exists(volume_id):
                    still_volumed.append(volume_id)

        clean = not still_running and not still_volumed
        detail = (
            "all pods and volumes torn down" if clean
            else f"pods still running: {still_running}; volumes still present: {still_volumed}"
        )
        return {"clean": clean, "detail": detail, "pods_checked": len(jobs)}
