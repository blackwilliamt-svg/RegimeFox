"""RunPod orchestration (spec 5): launch, poll, teardown verification.

No live RunPod calls, ever - every test substitutes a fake transport and
asserts on what was requested. This is the "no network access" principle the
rest of the suite already follows, applied to the one new external service.
"""
from __future__ import annotations

import pytest

from solbot.runpod import RunPodCapacityError, RunPodClient, RunPodError


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
        # gap-closure item 7: {gpu_type: $/hr}, and whether /gputypes itself
        # is reachable at all.
        self.gpu_prices: dict[str, float] = {}
        self.gputypes_fails: bool = False
        # gpu_type -> number of times POST /pods for that tier should raise
        # RunPodCapacityError before it is allowed to succeed - simulates
        # "no capacity right now" without touching the network at all.
        self.pod_capacity_fails: dict[str, int] = {}
        # When set, every POST /pods raises this instead - a non-capacity
        # failure (bad schema, bad auth, ...) that must not be retried or
        # trigger a tier fallback the way a capacity miss does.
        self.pod_error: Exception | None = None

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
            if self.pod_error is not None:
                raise self.pod_error
            gpu_type = (json.get("gpuTypeIds") or [None])[0]
            remaining = self.pod_capacity_fails.get(gpu_type, 0)
            if remaining > 0:
                self.pod_capacity_fails[gpu_type] = remaining - 1
                raise RunPodCapacityError(
                    "RunPod POST /pods returned HTTP 500: "
                    "could not find any pods with required specifications"
                )
            pod_id = self._id("pod")
            self.pods[pod_id] = {"id": pod_id, "status": "RUNNING", "gpuType": gpu_type}
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

        if method == "GET" and path == "/pods":
            return {"pods": list(self.pods.values())}

        raise RunPodError(f"unhandled fake request: {method} {path}")

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        # The only GraphQL query this codebase issues is gpuTypes - the GPU
        # catalog/pricing lookup, since RunPod's REST API has no equivalent
        # endpoint (a real key rotation attempt hitting exactly that gap -
        # a 400 from RunPod's own gateway for GET /v1/gputypes - is what
        # this fake exists to prevent regressing back to).
        self.calls.append(("POST", "graphql:gpuTypes", variables or {}))
        if self.gputypes_fails:
            raise RunPodError("simulated GraphQL gpuTypes outage")
        return {
            "gpuTypes": [
                {"id": name, "communityPrice": price}
                for name, price in self.gpu_prices.items()
            ]
        }


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def client(transport) -> RunPodClient:
    return RunPodClient(
        api_key="test-key", transport=transport,
        poll_interval_seconds=0.01, max_poll_seconds=0.05,
        capacity_retry_delay_seconds=0.0,
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
# Capacity errors: detection, retry, and GPU-tier fallback (fix-up: RunPod
# returns an HTTP 500 with "could not find any pods with required
# specifications" when no capacity is available for the requested tier -
# transient, not a bug, so the request is worth retrying, or trying a
# different tier, rather than failing the whole batch immediately.
# --------------------------------------------------------------------------
def test_create_pod_raises_the_capacity_specific_subclass(client, transport):
    transport.pod_capacity_fails["g"] = 1   # never clears - every attempt fails
    with pytest.raises(RunPodCapacityError):
        client.create_pod(name="x", gpu_type="g", volume_id="v", env={})


def test_a_non_capacity_error_is_not_a_capacity_error(client, transport):
    """A different failure (bad schema, bad auth, ...) must not be mistaken
    for a capacity miss - launch_batch's retry/fallback must not fire for it."""
    transport.pod_error = RunPodError(
        "RunPod POST /pods returned HTTP 400: malformed request"
    )
    with pytest.raises(RunPodError) as exc_info:
        client.create_pod(name="x", gpu_type="g", volume_id="v", env={})
    assert not isinstance(exc_info.value, RunPodCapacityError)


def _launch(client, transport, *, gpu_type="tier-a", fallback=(), mints=("AAA",)):
    return client.launch_batch(
        batch_index=0, mints=list(mints), candles=FakeCandles(), interval="1m",
        gpu_type=gpu_type, report_run_id=1, gpu_type_fallback=fallback,
    )


def test_launch_batch_retries_a_capacity_miss_and_then_succeeds(client, transport):
    # Fails twice, then succeeds on the third attempt - within the default
    # capacity_retry_attempts=3.
    transport.pod_capacity_fails["tier-a"] = 2

    job = _launch(client, transport)

    assert job["gpu_type"] == "tier-a"
    pod_calls = [c for c in transport.calls if c[0] == "POST" and c[1] == "/pods"]
    assert len(pod_calls) == 3   # two failures, one success


def test_launch_batch_does_not_retry_a_non_capacity_error(client, transport):
    """A schema/auth-shaped RunPodError must fail immediately, not eat three
    retries' worth of time on something retrying can never fix."""
    transport.pod_error = RunPodError(
        "RunPod POST /pods returned HTTP 400: malformed request"
    )

    with pytest.raises(RunPodError) as exc_info:
        _launch(client, transport)
    assert not isinstance(exc_info.value, RunPodCapacityError)

    pod_calls = [c for c in transport.calls if c[0] == "POST" and c[1] == "/pods"]
    assert len(pod_calls) == 1   # no retry at all


def test_launch_batch_falls_back_to_the_next_tier_when_the_preferred_one_is_full(client, transport):
    # The preferred tier never has capacity; the fallback tier succeeds
    # immediately.
    transport.pod_capacity_fails["tier-a"] = 999

    job = _launch(client, transport, fallback=("tier-b",))

    assert job["gpu_type"] == "tier-b"
    pod_calls = [c for c in transport.calls if c[0] == "POST" and c[1] == "/pods"]
    # capacity_retry_attempts (3) failed attempts on tier-a, then one
    # successful attempt on tier-b.
    assert len(pod_calls) == 4
    assert [c[2]["gpuTypeIds"][0] for c in pod_calls] == ["tier-a", "tier-a", "tier-a", "tier-b"]


def test_launch_batch_raises_clearly_when_every_tier_is_exhausted(client, transport):
    transport.pod_capacity_fails["tier-a"] = 999
    transport.pod_capacity_fails["tier-b"] = 999

    with pytest.raises(RunPodError) as exc_info:
        _launch(client, transport, fallback=("tier-b",))

    message = str(exc_info.value)
    assert "tier-a" in message
    assert "tier-b" in message


def test_launch_batch_cleans_up_the_volume_when_every_tier_is_exhausted(client, transport):
    transport.pod_capacity_fails["tier-a"] = 999

    with pytest.raises(RunPodError):
        _launch(client, transport)

    # The network volume created for the pod that never started must not be
    # left behind billing the operator for nothing.
    assert not transport.volumes


def test_launch_batch_with_no_fallback_configured_only_retries_the_one_tier(client, transport):
    """The GPU-tier benchmark's own call path - gpu_type_fallback left at
    its default empty tuple - must behave exactly like retry-only, no
    silent substitution of a different tier."""
    transport.pod_capacity_fails["tier-a"] = 999

    with pytest.raises(RunPodError) as exc_info:
        _launch(client, transport)

    assert "tier-a" in str(exc_info.value)
    pod_calls = [c for c in transport.calls if c[0] == "POST" and c[1] == "/pods"]
    assert len(pod_calls) == client.capacity_retry_attempts
    assert all(c[2]["gpuTypeIds"] == ["tier-a"] for c in pod_calls)


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


# --------------------------------------------------------------------------
# GPU-tier benchmark (gap-closure item 7)
# --------------------------------------------------------------------------
class FakeCandles:
    """A no-op stand-in for solbot.candlestore.ParquetCandleStore - the
    benchmark's own orchestration logic is what these tests pin, not the
    data-materialization path (already covered in test_wfmc.py)."""

    def materialize_bundle(self, mints, interval, out_dir):
        pass


def test_gpu_price_per_hour_reads_the_matching_tier(client, transport):
    transport.gpu_prices = {"NVIDIA RTX 4090": 0.79, "NVIDIA RTX 3090": 0.5}
    assert client.gpu_price_per_hour("NVIDIA RTX 4090") == 0.79
    assert client.gpu_price_per_hour("NVIDIA RTX 3090") == 0.5


def test_gpu_price_per_hour_uses_graphql_not_a_rest_path(client, transport):
    """Regression: RunPod's REST API has no GPU-types endpoint at all (a
    real key rotation hit a 400 from RunPod's own gateway for the
    previously-assumed GET /v1/gputypes) - the GPU catalog is GraphQL-only."""
    transport.gpu_prices = {"NVIDIA RTX 4090": 0.79}
    client.gpu_price_per_hour("NVIDIA RTX 4090")
    assert ("GET", "/gputypes", {}) not in transport.calls
    assert any(call[1] == "graphql:gpuTypes" for call in transport.calls)


def test_gpu_price_per_hour_is_none_for_an_unlisted_tier(client, transport):
    transport.gpu_prices = {"NVIDIA RTX 4090": 0.79}
    assert client.gpu_price_per_hour("NVIDIA H100") is None


def test_gpu_price_per_hour_is_none_when_the_catalog_is_unreachable(client, transport):
    transport.gputypes_fails = True
    assert client.gpu_price_per_hour("NVIDIA RTX 4090") is None


def test_benchmark_tier_reports_time_cost_and_clean_teardown(client, transport):
    transport.gpu_prices = {"NVIDIA RTX 4090": 0.79}

    # Wire the pod's self-termination to the poll count so verify_teardown's
    # wait actually observes it finish rather than force-stopping it.
    orig_create_pod = client.create_pod

    def create_pod_and_arm(*a, **kw):
        pod = orig_create_pod(*a, **kw)
        transport.pod_polls_until_stopped[pod["id"]] = 1
        return pod

    client.create_pod = create_pod_and_arm

    result = client.benchmark_tier(
        "NVIDIA RTX 4090", mints=["AAA", "BBB"], candles=FakeCandles(),
        interval="1m", report_run_id=1,
    )

    assert result.ok is True
    assert result.gpu_type == "NVIDIA RTX 4090"
    assert result.price_per_hour == 0.79
    assert result.cost_usd is not None and result.cost_usd >= 0
    assert result.teardown_clean is True
    # A self-exited pod's record can remain (stopped, not billing); its
    # volume must actually be gone.
    assert not transport.volumes
    assert all(p["status"] != "RUNNING" for p in transport.pods.values())


def test_benchmark_tier_still_tears_down_when_the_run_never_completes(client, transport):
    """A pod that never self-terminates within the poll window must still be
    force-stopped and its volume removed - the whole point of item 7's
    'teardown on every tier including a failure mid-benchmark' requirement."""
    transport.gpu_prices = {"NVIDIA RTX 4090": 0.79}
    orig_create_pod = client.create_pod

    def create_pod_never_stops(*a, **kw):
        pod = orig_create_pod(*a, **kw)
        transport.pod_polls_until_stopped[pod["id"]] = None
        return pod

    client.create_pod = create_pod_never_stops

    result = client.benchmark_tier(
        "NVIDIA RTX 4090", mints=["AAA"], candles=FakeCandles(), interval="1m", report_run_id=2,
    )

    assert result.ok is True   # the run itself launched and was waited on
    assert result.teardown_clean is True   # force-stopped by the safety net
    assert not transport.pods and not transport.volumes


def test_benchmark_tier_reports_failure_and_still_tears_down_on_a_launch_error(client, transport, monkeypatch):
    def broken_create_pod(*a, **kw):
        raise RunPodError("simulated launch failure")

    monkeypatch.setattr(client, "create_pod", broken_create_pod)

    result = client.benchmark_tier(
        "NVIDIA RTX 4090", mints=["AAA"], candles=FakeCandles(), interval="1m", report_run_id=3,
    )

    assert result.ok is False
    assert "simulated launch failure" in result.error
    assert result.cost_usd is None
    # The volume was created before the pod launch failed - it must still be
    # gone, even though there is no job dict for a pod that never existed.


def test_benchmark_tiers_continues_past_a_failed_tier(client, transport, monkeypatch):
    transport.gpu_prices = {"TIER-A": 1.0, "TIER-B": 2.0}
    original_create_pod = client.create_pod
    state = {"calls": 0}

    def flaky_create_pod(*a, **kw):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RunPodError("first tier fails to launch")
        pod = original_create_pod(*a, **kw)
        transport.pod_polls_until_stopped[pod["id"]] = 1
        return pod

    monkeypatch.setattr(client, "create_pod", flaky_create_pod)

    results = client.benchmark_tiers(
        ["TIER-A", "TIER-B"], mints=["AAA"], candles=FakeCandles(), interval="1m",
    )

    assert [r.gpu_type for r in results] == ["TIER-A", "TIER-B"]
    assert results[0].ok is False
    assert results[1].ok is True
    assert results[1].price_per_hour == 2.0


def test_benchmark_tier_uses_the_run_summary_lookup_when_given(client, transport):
    transport.gpu_prices = {"NVIDIA RTX 4090": 0.79}
    orig_create_pod = client.create_pod

    def create_pod_and_arm(*a, **kw):
        pod = orig_create_pod(*a, **kw)
        transport.pod_polls_until_stopped[pod["id"]] = 1
        return pod

    client.create_pod = create_pod_and_arm

    result = client.benchmark_tier(
        "NVIDIA RTX 4090", mints=["AAA"], candles=FakeCandles(), interval="1m",
        report_run_id=9, run_summary_lookup=lambda run_id: {"evaluated": 450},
    )

    assert result.combinations_evaluated == 450


def test_cost_per_1000_combinations_is_none_without_both_numbers():
    from solbot.runpod import BenchmarkResult

    assert BenchmarkResult("t", ok=True, cost_usd=1.0).cost_per_1000_combinations is None
    assert BenchmarkResult("t", ok=True, combinations_evaluated=10).cost_per_1000_combinations is None
    assert BenchmarkResult(
        "t", ok=True, cost_usd=2.0, combinations_evaluated=1000
    ).cost_per_1000_combinations == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Settings-page key validation ping (solbot.web.api._ping_provider)
# --------------------------------------------------------------------------
def test_ping_provider_validates_runpod_against_a_real_rest_endpoint(monkeypatch):
    """Regression: the settings page's "validate this RunPod key" check
    used to call GET /gputypes too, which RunPod's REST API does not have -
    every real RunPod key rotation failed with a 400 from RunPod's own
    gateway. GET /pods is a real, cheap, authenticated REST call."""
    from solbot.web.api import _ping_provider

    calls = []

    def fake_request(self, method, path, *, json=None):
        calls.append((method, path))
        return {"pods": []}

    monkeypatch.setattr("solbot.runpod.HttpxTransport.request", fake_request)

    ok, error = _ping_provider("runpod", "fake-key", config=None)

    assert ok is True
    assert error == ""
    assert ("GET", "/pods") in calls
    assert ("GET", "/gputypes") not in calls


def test_ping_provider_reports_the_real_failure_reason(monkeypatch):
    from solbot.runpod import RunPodError
    from solbot.web.api import _ping_provider

    def fake_request(self, method, path, *, json=None):
        raise RunPodError(
            "RunPod GET /gputypes returned HTTP 400: path does not exist in the specification"
        )

    monkeypatch.setattr("solbot.runpod.HttpxTransport.request", fake_request)

    ok, error = _ping_provider("runpod", "fake-key", config=None)

    assert ok is False
    assert "400" in error
