"""Expose a worker's live session stores through ``terrarium.session``.

The adapter supports history, search, store discovery, and adoption of a
``.kohakutr`` file that the controller has already copied to the worker.
"""

import os
from pathlib import Path
from typing import Any

from kohakuterrarium.laboratory._internal.app import AppMessage
from kohakuterrarium.laboratory.adapters.file_scopes import resolve_in_scope
from kohakuterrarium.laboratory.protocols import LabRegistrar
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.terrarium.engine import Terrarium
from kohakuterrarium.terrarium.graph_manifest import parse_manifest
from kohakuterrarium.terrarium.workspace_resume import (
    preflight_session_workspaces,
    WorkspaceResumeError,
    plan_workspace_resume,
    preflight_to_dict,
    preflight_workspace_resume,
)
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


class TerrariumSessionAdapter:
    """Worker-side ``terrarium.session`` APP extension."""

    NAMESPACE = "terrarium.session"

    def __init__(self, engine: Terrarium, lab_node: LabRegistrar) -> None:
        self._engine = engine
        self._node = lab_node
        lab_node.register_app_extension(self.NAMESPACE, self._dispatch)
        logger.info("lab adapter registered", namespace=self.NAMESPACE)

    def detach(self) -> None:
        self._node.unregister_app_extension(self.NAMESPACE)
        logger.info("lab adapter detached", namespace=self.NAMESPACE)

    async def _dispatch(self, msg: AppMessage) -> dict[str, Any]:
        try:
            return await self._handle(msg)
        except KeyError as e:
            return {"error": {"kind": "not_found", "message": str(e)}}
        except ValueError as e:
            return {"error": {"kind": "invalid", "message": str(e)}}
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("terrarium.session handler failed: %s", msg.type)
            return {"error": {"kind": "session", "message": str(e)}}

    async def _handle(self, msg: AppMessage) -> dict[str, Any]:
        match msg.type:
            case "history":
                return self._op_history(msg.body)
            case "search":
                return self._op_search(msg.body)
            case "stores":
                return self._op_stores(msg.body)
            case "workspace_preflight":
                return self._op_workspace_preflight(msg.body)
            case "resume":
                return await self._op_resume(msg.body)
            case "remove":
                return await self._op_remove(msg.body)
            case _:
                return {
                    "error": {
                        "kind": "unknown_type",
                        "message": f"unsupported terrarium.session type: {msg.type!r}",
                    }
                }

    def _op_history(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id = body.get("session_id")
        agent = body.get("agent")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        if not isinstance(agent, str) or not agent:
            raise ValueError("agent is required")
        store = self._resolve_store(session_id)
        events = store.get_events(agent)
        since = body.get("since")
        if isinstance(since, int):
            events = [e for e in events if int(e.get("event_id", 0)) > since]
        limit = body.get("limit")
        if isinstance(limit, int) and limit > 0:
            events = events[:limit]
        return {"events": events}

    def _op_search(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id = body.get("session_id")
        query = body.get("query")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        if not isinstance(query, str) or not query:
            raise ValueError("query is required")
        store = self._resolve_store(session_id)
        k = int(body.get("k") or 10)
        hits = store.search(query, k=k)
        return {"hits": hits}

    def _op_stores(self, body: dict[str, Any]) -> dict[str, Any]:
        # Only attached stores are authoritative for sessions owned by this worker.
        stores = getattr(self._engine, "_session_stores", {}) or {}
        return {"session_ids": sorted(stores.keys())}

    def _op_workspace_preflight(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate saved paths on the worker without creating a runtime."""
        raw_manifest = body.get("manifest")
        if raw_manifest is not None:
            manifest = parse_manifest(raw_manifest)
            replacements = body.get("workspace_overrides")
            pwd_override = body.get("pwd_override")
            if pwd_override is not None and replacements:
                raise ValueError(
                    "pwd_override and workspace_overrides are mutually exclusive"
                )
            if pwd_override is not None:
                replacements = {
                    item.creature_id: pwd_override for item in manifest.creatures
                }
            try:
                plan = plan_workspace_resume(
                    manifest,
                    replacements,
                    allow_valid_targets=pwd_override is not None,
                )
            except WorkspaceResumeError as exc:
                return {
                    "legacy": False,
                    "ready": False,
                    "members": [],
                    "gaps": [],
                    "error": {"code": exc.code.value, "message": str(exc)},
                }
            preflight = preflight_workspace_resume(plan.manifest)
            return {"legacy": False, **preflight_to_dict(preflight)}
        saved_pwd = body.get("legacy_pwd")
        if "legacy_pwd" in body:
            effective_pwd = body.get("pwd_override") or saved_pwd
            has_pwd = isinstance(effective_pwd, str) and bool(effective_pwd.strip())
            ready = has_pwd and Path(effective_pwd).is_dir()
            return {
                "legacy": True,
                "ready": ready,
                "members": [],
                "gaps": (
                    []
                    if ready
                    else [
                        {
                            "gap_id": "legacy",
                            "saved_pwd": (
                                effective_pwd if isinstance(effective_pwd, str) else ""
                            ),
                            "status": "invalid",
                            "creature_ids": [],
                        }
                    ]
                ),
            }
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("path, manifest, or legacy_pwd is required")
        local = Path(path)
        if not local.exists():
            raise FileNotFoundError(f"no .kohakutr at {path!r}")
        preflight = preflight_session_workspaces(local)
        if preflight is None:
            return {"legacy": True, "ready": True, "members": [], "gaps": []}
        return {"legacy": False, **preflight_to_dict(preflight)}

    async def _op_remove(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        creature_ids = [
            creature.creature_id
            for creature in self._engine.creatures.values()
            if creature.graph_id == session_id
        ]
        for creature_id in creature_ids:
            await self._engine.remove_creature(creature_id)
        return {"removed": creature_ids}

    async def _op_resume(self, body: dict[str, Any]) -> dict[str, Any]:
        """Adopt a session file already present on the worker."""
        scope = body.get("scope")
        rel = body.get("rel")
        if isinstance(scope, str) and scope:
            if not isinstance(rel, str) or not rel:
                raise ValueError("rel is required with scope")
            local = resolve_in_scope(scope, rel, self._engine)
        else:
            path = body.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError("path or scope+rel is required")
            local = Path(path)
        workspace_overrides = body.get("workspace_overrides")
        if body.get("pwd_override") is not None and workspace_overrides:
            raise ValueError(
                "pwd_override and workspace_overrides are mutually exclusive"
            )
        if not local.exists():
            raise FileNotFoundError(f"no .kohakutr at {str(local)!r}")
        sid = await self._engine.adopt_session(
            local,
            pwd=body.get("pwd_override"),
            workspace_overrides=workspace_overrides,
            llm=body.get("llm"),
        )
        store = getattr(self._engine, "_session_stores", {}).get(sid)
        meta = store.load_meta() if store is not None else {}
        # Path validity must be evaluated here; the controller cannot stat the
        # worker's filesystem.
        saved_pwd = str(meta.get("pwd", "") or "")
        return {
            "session_id": sid,
            "meta": dict(meta),
            "pwd_exists": (not saved_pwd) or os.path.isdir(saved_pwd),
        }

    def _resolve_store(self, session_id: str) -> SessionStore:
        stores = getattr(self._engine, "_session_stores", {}) or {}
        store = stores.get(session_id)
        if store is None:
            raise KeyError(f"no live session store for {session_id!r}")
        return store


__all__ = ["TerrariumSessionAdapter"]
