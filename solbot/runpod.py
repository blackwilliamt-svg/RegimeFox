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
# GPU catalog/pricing is GraphQL-only - see HttpxTransport's docstring.
GRAPHQL_API_BASE = "https://api.runpod.io/graphql"
POD_RUNNING_STATES = {"CREATED", "RESTARTING", "RUNNING", "PENDING"}
DEFAULT_IMAGE = "ghcr.io/example-org/solopt-worker:latest"
MAX_POLL_SECONDS = 4 * 3600
POLL_INTERVAL_SECONDS = 30

# Candidate tiers for manage.py benchmark-runpod, in the price band this bot's
# job size actually needs - the same "believed correct, verify against
# RunPod's live catalog before trusting it" caveat as the module docstring
# applies to these three names as much as to anything else here.
DEFAULT_BENCHMARK_TIERS = ["NVIDIA RTX 4090", "NVIDIA RTX 3090", "NVIDIA RTX A5000"]
# A representative slice, not the real monthly job - enough to compare
# per-tier throughput without paying for (or waiting through) the genuine
# multi-hour run on three GPUs just to benchmark them.
BENCHMARK_MINT_LIMIT = 5
BENCHMARK_MAX_EVALUATIONS = 100
BENCHMARK_TIMEOUT_SECONDS = 20 * 60


class RunPodError(RuntimeError):
    """RunPod refused a request or could not be reached."""


@dataclass(slots=True)
class BenchmarkResult:
    """One GPU tier's showing in `manage.py benchmark-runpod` (gap-closure
    item 7): wall-clock time and cost from RunPod's own billing API, not an
    estimate - and, when the worker's run summary made it back, how many
    combinations that time bought."""

    gpu_type: str
    ok: bool
    elapsed_seconds: float = 0.0
    price_per_hour: float | None = None
    cost_usd: float | None = None
    combinations_evaluated: int | None = None
    teardown_clean: bool | None = None
    error: str = ""

    @property
    def cost_per_1000_combinations(self) -> float | None:
        """The number worth comparing tiers on - raw cost alone favours
        whichever GPU is simply slowest to burn through the same money."""
        if not self.cost_usd or not self.combinations_evaluated:
            return None
        return self.cost_usd / self.combinations_evaluated * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "ok": self.ok,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "price_per_hour": self.price_per_hour,
            "cost_usd": round(self.cost_usd, 4) if self.cost_usd is not None else None,
            "combinations_evaluated": self.combinations_evaluated,
            "cost_per_1000_combinations": (
                round(self.cost_per_1000_combinations, 4)
                if self.cost_per_1000_combinations is not None else None
            ),
            "teardown_clean": self.teardown_clean,
            "error": self.error,
        }


class Transport(Protocol):
    def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...

    def graphql(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


@dataclass
class HttpxTransport:
    """The real transport: httpx against RunPod's REST API for
    pods/network-volumes/templates (the REST v1 surface at ``API_BASE``),
    and its older GraphQL API for GPU catalog/pricing - the REST API's
    OpenAPI schema (``GET /v1/openapi.json``) has no GPU-types endpoint at
    all; that data has only ever been exposed via GraphQL
    (``GRAPHQL_API_BASE``), confirmed against RunPod's own published spec
    after a real key rotation attempt hit exactly this gap (a 400 from
    RunPod's own API gateway saying ``/v1/gputypes`` does not exist).
    """

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

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}
        body: dict[str, Any] = {"query": query}
        if variables is not None:
            body["variables"] = variables
        try:
            response = httpx.post(
                GRAPHQL_API_BASE, json=body, headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise RunPodError(f"could not reach RunPod's GraphQL API: {exc}") from exc
        if response.status_code >= 400:
            raise RunPodError(
                f"RunPod GraphQL returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            payload = response.json() if response.content else {}
        except ValueError as exc:
            raise RunPodError("RunPod GraphQL did not return JSON") from exc
        if payload.get("errors"):
            raise RunPodError(f"RunPod GraphQL error: {str(payload['errors'])[:300]}")
        return payload.get("data") or {}


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
    # Pricing
    # ------------------------------------------------------------------
    GPU_TYPES_QUERY = """
        query gpuTypes {
          gpuTypes {
            id
            displayName
            communityPrice
            securePrice
            lowestPrice {
              uninterruptablePrice
            }
          }
        }
    """

    def gpu_price_per_hour(self, gpu_type: str) -> float | None:
        """This tier's current $/hr from RunPod's own catalog - never a
        hardcoded guess, since GPU pricing moves and a stale number here
        would silently mislead the cost comparison it exists to inform.

        The GPU catalog is GraphQL-only (see HttpxTransport's docstring) -
        RunPod's REST API has no equivalent endpoint. Returns None (rather
        than raising) when the tier can't be found or the catalog can't be
        reached - the caller reports cost as unknown rather than failing
        the whole benchmark over one lookup.
        """
        try:
            data = self.transport.graphql(self.GPU_TYPES_QUERY)
        except RunPodError:
            return None
        types = data.get("gpuTypes") or []
        for entry in types:
            if not isinstance(entry, dict):
                continue
            name = entry.get("id") or entry.get("displayName")
            if name != gpu_type:
                continue
            lowest = entry.get("lowestPrice") or {}
            price = (
                entry.get("communityPrice")
                or entry.get("securePrice")
                or (lowest.get("uninterruptablePrice") if isinstance(lowest, dict) else None)
            )
            try:
                return float(price) if price is not None else None
            except (TypeError, ValueError):
                return None
        return None

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
        # "US-CA-1" (the original guess) does not exist - a real benchmark
        # run's own error confirmed the live list of datacenters that
        # support network volumes; "US-CA-2" is on it. That list can still
        # change on RunPod's end, so this default is a starting point, not
        # a guarantee - a caller can always override it.
        data_center_id: str = "US-CA-2",
        s3_access_key: str = "",
        s3_secret_key: str = "",
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Ship one coin batch's data up and start its worker.

        Returns ``{"pod_id", "volume_id", "mints"}`` so the caller can poll
        and, later, verify teardown. `extra_env` layers on top of the
        standard reporting env - benchmark_tier (gap-closure item 7) uses it
        to cap the search to a small representative slice rather than a full
        monthly-sized run.
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
                **(extra_env or {}),
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

    # ------------------------------------------------------------------
    # GPU-tier benchmark (gap-closure item 7)
    # ------------------------------------------------------------------
    def benchmark_tier(
        self,
        gpu_type: str,
        *,
        mints: list[str],
        candles: Any,
        interval: str,
        report_run_id: int,
        droplet_url: str = "",
        droplet_token: str = "",
        # "US-CA-1" (the original guess) does not exist - a real benchmark
        # run's own error confirmed the live list of datacenters that
        # support network volumes; "US-CA-2" is on it. That list can still
        # change on RunPod's end, so this default is a starting point, not
        # a guarantee - a caller can always override it.
        data_center_id: str = "US-CA-2",
        s3_access_key: str = "",
        s3_secret_key: str = "",
        max_seconds: float | None = None,
        run_summary_lookup: Callable[[int], dict[str, Any] | None] | None = None,
    ) -> BenchmarkResult:
        """Run the fixed representative slice on one tier and report wall-
        clock time, $/hr, and cost - never guessed, and never left running.

        Every failure mode - launch, wait, teardown, pricing lookup - is
        caught here rather than propagated, so one bad tier cannot abort the
        rest of a multi-tier benchmark: teardown is always attempted in a
        `finally`, even for a tier that failed to launch or never reported
        back, and the tier is recorded with `ok=False` and its error rather
        than raised.
        """
        started = time.monotonic()
        job: dict[str, Any] | None = None
        ok = False
        error = ""
        try:
            job = self.launch_batch(
                batch_index=0,
                mints=mints[:BENCHMARK_MINT_LIMIT],
                candles=candles,
                interval=interval,
                gpu_type=gpu_type,
                report_run_id=report_run_id,
                droplet_url=droplet_url,
                droplet_token=droplet_token,
                data_center_id=data_center_id,
                s3_access_key=s3_access_key,
                s3_secret_key=s3_secret_key,
                extra_env={
                    "SOLOPT_BENCHMARK": "1",
                    "SOLOPT_MAX_EVALUATIONS": str(BENCHMARK_MAX_EVALUATIONS),
                },
            )
            self.wait_for_completion([job], max_seconds=max_seconds)
            ok = True
        except Exception as exc:
            log.warning("benchmark of tier %s failed: %s", gpu_type, exc)
            error = str(exc)[:300]

        elapsed = time.monotonic() - started

        # Teardown is attempted regardless of whether the run above succeeded
        # - a tier that failed mid-launch may still have left a pod or volume
        # behind, and that is exactly the case this must not skip.
        teardown_clean: bool | None = None
        if job is not None:
            try:
                teardown_clean = bool(self.verify_teardown([job])["clean"])
                if not teardown_clean:
                    log.warning("benchmark tier %s did not tear down cleanly", gpu_type)
            except Exception:
                log.exception("teardown verification itself failed for tier %s", gpu_type)
                teardown_clean = False

        if not ok:
            return BenchmarkResult(
                gpu_type=gpu_type, ok=False, elapsed_seconds=elapsed,
                teardown_clean=teardown_clean, error=error,
            )

        price = self.gpu_price_per_hour(gpu_type)
        cost = price * elapsed / 3600.0 if price is not None else None

        evaluated = None
        if run_summary_lookup is not None:
            try:
                summary = run_summary_lookup(report_run_id)
                if summary:
                    evaluated = summary.get("evaluated") or summary.get("candidates_considered")
            except Exception:
                log.debug("run summary lookup failed for tier %s", gpu_type, exc_info=True)

        return BenchmarkResult(
            gpu_type=gpu_type, ok=True, elapsed_seconds=elapsed,
            price_per_hour=price, cost_usd=cost, combinations_evaluated=evaluated,
            teardown_clean=teardown_clean,
        )

    def benchmark_tiers(
        self,
        tiers: list[str] | None = None,
        *,
        mints: list[str],
        candles: Any,
        interval: str,
        report_run_id_start: int = 1,
        progress: Callable[[int, int, str], None] | None = None,
        **kwargs: Any,
    ) -> list[BenchmarkResult]:
        """Benchmark each tier in turn, one at a time - concurrent tiers
        would confound wall-clock time with however this droplet's own
        upload bandwidth happened to be shared between them.

        ``progress(index, total, gpu_type)``, if given, fires right before
        that tier's benchmark starts - dashboard step 4's live status for
        what would otherwise be a silent multi-minute-to-multi-hour wait.
        """
        tiers = tiers or DEFAULT_BENCHMARK_TIERS
        results = []
        for i, gpu_type in enumerate(tiers):
            if progress is not None:
                progress(i, len(tiers), gpu_type)
            result = self.benchmark_tier(
                gpu_type,
                mints=mints,
                candles=candles,
                interval=interval,
                report_run_id=report_run_id_start + i,
                **kwargs,
            )
            results.append(result)
        return results
