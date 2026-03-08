from fastapi.testclient import TestClient

from trenches_env.training_server import NoOpSourceHarvester, create_training_app


def test_noop_source_harvester_returns_pending_packets() -> None:
    harvester = NoOpSourceHarvester()

    training_packets, live_packets = harvester.get_packets_for_agent("us", include_live=True)

    assert training_packets
    assert live_packets
    assert all(packet.status == "pending" for packet in training_packets)
    assert all(packet.status == "pending" for packet in live_packets)
    assert harvester.refresh_due_batch(include_live=True) == 0


def test_training_app_exposes_healthz() -> None:
    with TestClient(create_training_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
