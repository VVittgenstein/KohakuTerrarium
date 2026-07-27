"""Remote-node resume path tests for :mod:`api.routes.persistence.resume`."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from kohakuterrarium.api.deps import (
    get_service_factory,
    resolve_request_session_dir,
)
from kohakuterrarium.api.routes.persistence import resume as resume_mod
from kohakuterrarium.api.routes.persistence import resume_remote as remote_mod
from kohakuterrarium.studio.sessions.registry import meta_for


@pytest.fixture(autouse=True)
def _workspace_resume_boundaries(monkeypatch):
    async def ready(*args, **kwargs):
        return {"legacy": False, "ready": True, "members": [], "gaps": []}

    monkeypatch.setattr(resume_mod, "_worker_workspace_preflight", ready)
    monkeypatch.setattr(
        resume_mod,
        "_persist_remote_workspace_meta",
        lambda path, *args, **kwargs: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        resume_mod, "_rollback_remote_workspace_meta", lambda snapshot: None
    )


def _app(*, engine=None, service=None):
    app = FastAPI()
    resolved = service or engine or SimpleNamespace()
    app.dependency_overrides[get_service_factory] = lambda: lambda: resolved
    app.dependency_overrides[resolve_request_session_dir] = lambda: Path("/")
    app.include_router(resume_mod.router, prefix="/sessions")
    return app


class _FakeHost:
    # The resume route pushes the ``.kohakutr`` via the chunked
    # ``write_stream`` family (write_begin / write_chunk / write_commit)
    # rather than a one-shot ``write`` — a session file routinely
    # exceeds the Lab transport frame ceiling. Default responses make
    # that handshake succeed; a test overrides them to exercise a
    # specific failure.
    _STREAM_DEFAULTS = {
        "terrarium.files:write_begin": {
            "transfer_id": "tid-1",
            "chunk_size": 262144,
        },
        "terrarium.files:write_chunk": {"received": 0},
        "terrarium.files:write_commit": {"written": 0, "sha256": ""},
    }

    def __init__(self, responses=None, raises=None):
        self._responses = dict(self._STREAM_DEFAULTS)
        self._responses.update(responses or {})
        self._raises = raises or {}
        self.calls = []

    async def request(self, *, to_node, namespace, type, body, timeout):
        self.calls.append(
            {"namespace": namespace, "type": type, "to": to_node, "body": body}
        )
        key = f"{namespace}:{type}"
        if key in self._raises:
            raise self._raises[key]
        return self._responses.get(key, {})


class _Svc:
    def __init__(self, host, nodes=("w1",)):
        self.host = host
        self._nodes = nodes

    def connected_nodes(self):
        return tuple(self._nodes)


class TestRemoteWritePath:
    @pytest.mark.asyncio
    async def test_worker_preflight_rejects_partial_dirty_without_rpc(
        self, monkeypatch, tmp_path
    ):
        path = tmp_path / "dirty.kohakutr"
        path.write_bytes(b"data")
        monkeypatch.setattr(
            remote_mod,
            "read_session_meta",
            lambda _path: {
                "workspace_resume_state": {"status": "partial_dirty"},
                "live_graph_manifest": {},
            },
        )
        host = _FakeHost()

        with pytest.raises(HTTPException) as raised:
            await remote_mod.worker_workspace_preflight(host, path, "w1")

        assert raised.value.status_code == 409
        assert raised.value.detail["code"] == "partial_dirty"
        assert host.calls == []

    def test_file_read_error(self, monkeypatch, tmp_path):
        # resolve_session_path_default returns a path that doesn't exist.
        ghost = tmp_path / "missing.kohakutr"
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: ghost
        )
        svc = _Svc(_FakeHost())
        client = TestClient(_app(service=svc))
        resp = client.post("/sessions/x/resume", json={"on_node": "w1"})
        assert resp.status_code == 404

    def test_write_response_error(self, monkeypatch, tmp_path):
        p = tmp_path / "x.kohakutr"
        p.write_bytes(b"data")
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: p
        )
        host = _FakeHost(
            responses={"terrarium.files:stat": {"error": {"message": "no write"}}}
        )
        client = TestClient(_app(service=_Svc(host)))
        resp = client.post("/sessions/x/resume", json={"on_node": "w1"})
        assert resp.status_code == 502

    def test_resume_response_error(self, monkeypatch, tmp_path):
        p = tmp_path / "x.kohakutr"
        p.write_bytes(b"data")
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: p
        )
        host = _FakeHost(
            responses={
                "terrarium.files:stat": {"ok": True},
                "terrarium.session:resume": {"error": {"message": "bad resume"}},
            }
        )
        client = TestClient(_app(service=_Svc(host)))
        resp = client.post("/sessions/x/resume", json={"on_node": "w1"})
        assert resp.status_code == 502

    def test_transport_exception(self, monkeypatch, tmp_path):
        # The chunked push raises mid-handshake (write_begin) — the
        # route must surface a clean 502, not propagate the error.
        p = tmp_path / "x.kohakutr"
        p.write_bytes(b"data")
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: p
        )
        host = _FakeHost(
            raises={"terrarium.files:write_begin": RuntimeError("transport down")}
        )
        client = TestClient(_app(service=_Svc(host)))
        resp = client.post("/sessions/x/resume", json={"on_node": "w1"})
        assert resp.status_code == 502

    def test_remote_success(self, monkeypatch, tmp_path):
        p = tmp_path / "x.kohakutr"
        p.write_bytes(b"data")
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: p
        )
        host = _FakeHost(
            responses={
                "terrarium.files:stat": {"ok": True},
                "terrarium.session:resume": {
                    "session_id": "remote-sid",
                    "meta": {
                        "config_type": "terrarium",
                        "terrarium_name": "remote-t",
                        "agents": ["alice", "bob"],
                        "pwd": "/p",
                        "terrarium_creatures": [{"name": "x"}],
                    },
                },
            }
        )
        client = TestClient(_app(service=_Svc(host)))
        resp = client.post("/sessions/x/resume", json={"on_node": "w1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["instance_id"] == "remote-sid"
        assert body["type"] == "terrarium"
        assert body["on_node"] == "w1"
        # The push went through the chunked write_stream handshake, not
        # a one-shot ``write`` — that is the whole point of the pack
        # system: no single APP message can overflow the transport.
        pushed = [c["type"] for c in host.calls if c["namespace"] == "terrarium.files"]
        assert "write_begin" in pushed
        assert "write_commit" in pushed
        assert "write" not in pushed

    def test_pwd_override_threads_to_worker(self, monkeypatch, tmp_path):
        p = tmp_path / "x.kohakutr"
        p.write_bytes(b"data")
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: p
        )
        host = _FakeHost(
            responses={
                "terrarium.files:stat": {"ok": True},
                "terrarium.session:resume": {
                    "session_id": "sid",
                    "meta": {"agents": ["a"], "pwd": "/p"},
                },
            }
        )
        client = TestClient(_app(service=_Svc(host)))
        resp = client.post(
            "/sessions/x/resume", json={"on_node": "w1", "pwd": "/new/dir"}
        )
        assert resp.status_code == 200
        resume_calls = [c for c in host.calls if c["namespace"] == "terrarium.session"]
        assert resume_calls[0]["body"]["pwd_override"] == "/new/dir"
        assert resume_calls[0]["body"]["scope"] == "config://"
        assert resume_calls[0]["body"]["rel"] == "resume/x.kohakutr"

    def test_agent_response_preserves_legacy_type_name_and_fallback(
        self, monkeypatch, tmp_path
    ):
        p = tmp_path / "x.kohakutr"
        p.write_bytes(b"data")
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: p
        )
        host = _FakeHost(
            responses={
                "terrarium.files:stat": {"ok": True},
                "terrarium.session:resume": {
                    "session_id": "sid",
                    "meta": {
                        "config_type": "agent",
                        "terrarium_name": "saved-name",
                        "agents": ["alice"],
                        "pwd": "",
                    },
                },
            }
        )

        response = TestClient(_app(service=_Svc(host))).post(
            "/sessions/x/resume", json={"on_node": "w1"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "agent"
        assert body["session_name"] == "saved-name"
        assert body["session"]["creatures"] == [
            {"creature_id": "alice", "name": "alice"}
        ]
        assert body["session"]["pwd_exists"] is True

    def test_worker_pwd_exists_overrides_controller_stat(self, monkeypatch, tmp_path):
        # meta pwd "" makes the controller-side __post_init__ compute
        # True — but the session lives on the worker, whose report of a
        # missing dir must win.
        p = tmp_path / "x.kohakutr"
        p.write_bytes(b"data")
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: p
        )
        host = _FakeHost(
            responses={
                "terrarium.files:stat": {"ok": True},
                "terrarium.session:resume": {
                    "session_id": "sid",
                    "meta": {"agents": ["a"], "pwd": ""},
                    "pwd_exists": False,
                },
            }
        )
        client = TestClient(_app(service=_Svc(host)))
        resp = client.post("/sessions/x/resume", json={"on_node": "w1"})
        assert resp.status_code == 200
        assert resp.json()["session"]["pwd_exists"] is False

    def test_remote_no_session_id_502(self, monkeypatch, tmp_path):
        p = tmp_path / "x.kohakutr"
        p.write_bytes(b"data")
        monkeypatch.setattr(
            resume_mod, "resolve_session_path_in", lambda n, _session_dir: p
        )
        host = _FakeHost(
            responses={
                "terrarium.files:stat": {"ok": True},
                "terrarium.session:resume": {"meta": {}},
            }
        )
        client = TestClient(_app(service=_Svc(host)))
        resp = client.post("/sessions/x/resume", json={"on_node": "w1"})
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_response_build_failure_restores_only_its_registry_entries(
        self, monkeypatch, tmp_path
    ):
        service = _Svc(_FakeHost())
        service._home = {"existing": "w1"}

        async def list_creatures():
            return [
                SimpleNamespace(
                    creature_id="alice",
                    name="alice",
                    graph_id="sid",
                    is_running=True,
                    is_privileged=False,
                )
            ]

        service.list_creatures = list_creatures
        registry = meta_for(service)
        registry["sid"] = {"name": "previous"}
        registry["other"] = {"name": "untouched"}

        class FailingSession:
            def __init__(self, **kwargs):
                raise RuntimeError("synthetic response failed")

        monkeypatch.setattr(remote_mod, "Session", FailingSession)

        with pytest.raises(RuntimeError, match="synthetic response failed"):
            await remote_mod.build_remote_response(
                service,
                sid="sid",
                meta={"agents": ["alice"]},
                on_node="w1",
                path=tmp_path / "saved.kohakutr",
                session_name="saved",
                worker_pwd_exists=None,
            )

        assert service._home == {"existing": "w1"}
        assert registry == {
            "sid": {"name": "previous"},
            "other": {"name": "untouched"},
        }


# ---------------------------------------------------------------------------
# CF-6 — cluster resume
# ---------------------------------------------------------------------------


class _ClusterSvc(_Svc):
    """``_Svc`` extended with a recording ``connect`` so the cluster
    test can assert the relink fires once per non-primary member."""

    def __init__(self, host, nodes=("w1", "w2")):
        super().__init__(host, nodes=nodes)
        self.connect_calls: list[tuple[str, str]] = []
        # Mirror the home registry the resume route updates.  Roster
        # entries are shaped to mimic ``CreatureInfo``.
        self._roster: list = []
        self.disconnect_calls: list[tuple[str, str, str | None]] = []

    async def connect(self, sender_id, receiver_id, *, channel=None):
        self.connect_calls.append((sender_id, receiver_id))
        return SimpleNamespace(channel=channel or "auto", graph_id=sender_id)

    async def disconnect(self, sender_id, receiver_id, *, channel=None):
        self.disconnect_calls.append((sender_id, receiver_id, channel))
        return SimpleNamespace(channels=[channel] if channel else [])

    async def list_creatures(self):
        return tuple(self._roster)


def _ci(creature_id, name, graph_id, *, is_running=True):
    """Build a minimal stand-in for ``CreatureInfo``."""
    return SimpleNamespace(
        creature_id=creature_id,
        name=name,
        graph_id=graph_id,
        is_running=is_running,
        is_privileged=True,
    )


class TestClusterResume:
    """CF-6 — multi-worker cluster resume.

    On request body ``{members: [{sid, on_node}, ...]}``: each member's
    ``.kohakutr`` is pushed to its own worker, every worker resume RPC
    fires, and ``service.connect()`` is invoked between the primary
    creature and every peer so ``_cluster_links`` is repopulated.
    """

    def test_cluster_resume_pushes_to_every_worker_and_relinks(
        self, monkeypatch, tmp_path
    ):
        # Two saved mirror files — one per member.  Names match each
        # member's sid so the route's per-member path resolution lands
        # them deterministically.
        sid_a, sid_b = "sid-a", "sid-b"
        pa = tmp_path / f"{sid_a}.kohakutr"
        pb = tmp_path / f"{sid_b}.kohakutr"
        pa.write_bytes(b"alpha")
        pb.write_bytes(b"bravo")
        paths = {sid_a: pa, sid_b: pb}
        monkeypatch.setattr(
            resume_mod,
            "resolve_session_path_in",
            lambda name, _session_dir: paths.get(name),
        )
        # Worker resume RPCs each return a fresh new sid + meta carrying
        # the agents list. Per-worker dispatch is keyed on `to_node`.
        per_node_resume = {
            "w1": {
                "session_id": "new-a",
                "meta": {"agents": ["alpha"], "config_type": "agent"},
            },
            "w2": {
                "session_id": "new-b",
                "meta": {"agents": ["bravo"], "config_type": "agent"},
            },
        }

        class _RoutedHost(_FakeHost):
            async def request(self, *, to_node, namespace, type, body, timeout):
                self.calls.append(
                    {
                        "namespace": namespace,
                        "type": type,
                        "to": to_node,
                        "body": body,
                    }
                )
                if namespace == "terrarium.session" and type == "resume":
                    return per_node_resume[to_node]
                if namespace == "terrarium.files" and type == "stat":
                    return {"ok": True}
                return self._responses.get(f"{namespace}:{type}", {})

        host = _RoutedHost()
        svc = _ClusterSvc(host)
        svc._roster = [
            _ci("cid-alpha", "alpha", "new-a"),
            _ci("cid-alpha-2", "alpha-2", "new-a"),
            _ci("cid-bravo", "bravo", "new-b"),
        ]
        client = TestClient(_app(service=svc))
        resp = client.post(
            f"/sessions/{sid_a}/resume",
            json={
                "on_node": "w1",
                "members": [
                    {"sid": sid_a, "on_node": "w1"},
                    {"sid": sid_b, "on_node": "w2"},
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Both workers were targeted.  Each got its own resume RPC.
        resume_targets = sorted(
            c["to"]
            for c in host.calls
            if c["namespace"] == "terrarium.session" and c["type"] == "resume"
        )
        assert resume_targets == ["w1", "w2"]
        # The host relinked the cluster via service.connect between the
        # primary creature and the peer — this is the CF-6 fix:
        # without it, _cluster_links would stay empty after resume and
        # the cluster would silently downgrade to two singletons.
        assert svc.connect_calls == [("cid-alpha", "cid-bravo")]
        # Response surfaces the resumed primary and the cluster_members
        # list (so the frontend knows the cluster is intact).
        assert body["instance_id"] == "new-a"
        assert [item["creature_id"] for item in body["session"]["creatures"]] == [
            "cid-alpha",
            "cid-alpha-2",
        ]
        assert {m["on_node"] for m in body["cluster_members"]} == {"w1", "w2"}

    def test_cluster_resume_rejects_when_member_worker_disconnected(
        self, monkeypatch, tmp_path
    ):
        # Saved mirrors exist but w2 is NOT in connected_nodes.
        sid_a, sid_b = "sid-a", "sid-b"
        pa = tmp_path / f"{sid_a}.kohakutr"
        pa.write_bytes(b"alpha")
        monkeypatch.setattr(
            resume_mod,
            "resolve_session_path_in",
            lambda name, _session_dir: pa,
        )
        host = _FakeHost()
        svc = _ClusterSvc(host, nodes=("w1",))  # w2 missing on purpose
        client = TestClient(_app(service=svc))
        resp = client.post(
            f"/sessions/{sid_a}/resume",
            json={
                "on_node": "w1",
                "members": [
                    {"sid": sid_a, "on_node": "w1"},
                    {"sid": sid_b, "on_node": "w2"},
                ],
            },
        )
        # Behavior: half-resumed clusters are worse than an honest 404,
        # so the route must reject upfront when any member's worker is
        # absent — no file is pushed, no connect is called.
        assert resp.status_code == 404
        assert svc.connect_calls == []
        pushed = [c for c in host.calls if c["namespace"] == "terrarium.files"]
        assert pushed == []

    def test_cluster_relink_failure_disconnects_completed_links(
        self, monkeypatch, tmp_path
    ):
        members = [
            ("sid-a", "w1", "new-a", "cid-a"),
            ("sid-b", "w2", "new-b", "cid-b"),
            ("sid-c", "w3", "new-c", "cid-c"),
        ]
        paths = {}
        for sid, _node, _new_sid, _cid in members:
            path = tmp_path / f"{sid}.kohakutr"
            path.write_bytes(sid.encode())
            paths[sid] = path
        monkeypatch.setattr(
            resume_mod,
            "resolve_session_path_in",
            lambda name, _session_dir: paths.get(name),
        )
        resumed_by_node = {
            node: {
                "session_id": new_sid,
                "meta": {"agents": [cid], "config_type": "agent"},
            }
            for _sid, node, new_sid, cid in members
        }

        class _RoutedHost(_FakeHost):
            async def request(self, *, to_node, namespace, type, body, timeout):
                self.calls.append(
                    {
                        "namespace": namespace,
                        "type": type,
                        "to": to_node,
                        "body": body,
                    }
                )
                if namespace == "terrarium.session" and type == "resume":
                    return resumed_by_node[to_node]
                if namespace == "terrarium.session" and type == "remove":
                    return {}
                if namespace == "terrarium.files" and type == "stat":
                    return {"ok": True}
                return self._responses.get(f"{namespace}:{type}", {})

        class _FailingService(_ClusterSvc):
            async def connect(self, sender_id, receiver_id, *, channel=None):
                self.connect_calls.append((sender_id, receiver_id))
                if receiver_id == "cid-c":
                    raise RuntimeError("relink failed")
                return SimpleNamespace(channel="resume-link", graph_id=sender_id)

        host = _RoutedHost()
        service = _FailingService(host, nodes=("w1", "w2", "w3"))
        service._roster = [
            _ci(cid, cid, new_sid) for _sid, _node, new_sid, cid in members
        ]
        client = TestClient(_app(service=service))

        response = client.post(
            "/sessions/sid-a/resume",
            json={
                "on_node": "w1",
                "members": [
                    {"sid": sid, "on_node": node}
                    for sid, node, _new_sid, _cid in members
                ],
            },
        )

        assert response.status_code == 502
        assert service.disconnect_calls == [("cid-a", "cid-b", "resume-link")]
        removed = [
            (call["to"], call["body"]["session_id"])
            for call in host.calls
            if call["namespace"] == "terrarium.session" and call["type"] == "remove"
        ]
        assert sorted(removed) == [
            ("w1", "new-a"),
            ("w2", "new-b"),
            ("w3", "new-c"),
        ]
