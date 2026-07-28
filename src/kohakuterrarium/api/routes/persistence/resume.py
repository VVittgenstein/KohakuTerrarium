"""Persistence resume — adopt a saved session into the live engine.

The ``/{session_name}/resume`` path allows this router to mount under
``/api/sessions`` while preserving the frontend session API URL.

Responses retain ``{instance_id, type, session_name}`` and include the full
:class:`Session` handle under ``session``.

In standalone mode, ``on_node="_host"`` resumes into the local engine. In
lab-host mode, a worker target receives the ``.kohakutr`` file and adopts it
through ``terrarium.session.resume``; the host itself does not run agents.
"""

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from kohakuterrarium.api.deps import (
    get_service_factory,
    resolve_request_session_dir,
)
from kohakuterrarium.api.routes.persistence.resume_remote import (
    build_remote_response as _build_remote_response,
    build_cluster_response as _build_cluster_response,
    cleanup_remote_sessions as _cleanup_remote_sessions,
    RemoteMirrorRollbackError,
    persist_remote_workspace_meta as _persist_remote_workspace_meta,
    push_and_resume_member as _push_and_resume_member,
    register_cluster_members as _register_cluster_members,
    rollback_remote_workspace_meta as _rollback_remote_workspace_meta,
    unregister_cluster_members as _unregister_cluster_members,
    worker_workspace_preflight as _worker_workspace_preflight,
)
from kohakuterrarium.session.readonly import read_session_meta
from kohakuterrarium.errors import SessionNotResumableError
from kohakuterrarium.studio.persistence.resume import resume_session as studio_resume
from kohakuterrarium.terrarium.graph_manifest import MANIFEST_KEY, parse_manifest
from kohakuterrarium.terrarium.workspace_resume import (
    WorkspaceResumeError,
    plan_workspace_resume,
    preflight_session_workspaces,
    preflight_to_dict,
    preflight_workspace_resume,
)
from kohakuterrarium.studio.persistence.store import resolve_session_path_in
from kohakuterrarium.studio.persistence.viewer.paths import normalize_session_stem
from kohakuterrarium.terrarium.resume import prepare_resume_workspace
from kohakuterrarium.terrarium.service import TerrariumService

router = APIRouter()


class ClusterMember(BaseModel):
    """One member of a cluster session."""

    sid: str
    on_node: str


class WorkspacePreflightRequest(BaseModel):
    """Optional execution node and candidate workspace replacements."""

    on_node: str = "_host"
    members: list[ClusterMember] | None = None
    pwd: str | None = None
    workspace_overrides: dict[str, str] | None = None
    member_workspace_overrides: dict[str, dict[str, str]] | None = None
    member_pwd_overrides: dict[str, str] | None = None


@router.post("/{session_name}/resume/preflight")
async def preflight_resume(
    session_name: str,
    req: WorkspacePreflightRequest | None = None,
    session_dir: Path = Depends(resolve_request_session_dir),
    service_factory: Callable[[], TerrariumService] = Depends(get_service_factory),
) -> dict[str, Any]:
    """Inspect local session workspaces without creating a runtime."""
    if req is not None and req.pwd is not None and req.workspace_overrides:
        raise HTTPException(
            status_code=422,
            detail="pwd and workspace_overrides are mutually exclusive",
        )
    path = await asyncio.to_thread(resolve_session_path_in, session_name, session_dir)
    if path is None:
        raise HTTPException(status_code=404, detail="Session not found")
    cluster_members = (
        req.members if req is not None and req.members else None
    ) or await asyncio.to_thread(_read_saved_cluster_members, path)
    if cluster_members:
        if req is not None and req.workspace_overrides and len(cluster_members) > 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Cluster workspace overrides must be scoped by member session ID"
                ),
            )
        service = service_factory()
        host = getattr(service, "host", None)
        if host is None:
            raise HTTPException(status_code=503, detail="Lab host not configured")
        results = []
        for member in cluster_members:
            member_path = await asyncio.to_thread(
                resolve_session_path_in, member.sid, session_dir
            )
            if member_path is None:
                raise HTTPException(
                    status_code=404, detail=f"Session {member.sid!r} not found"
                )
            replacements = (req.member_workspace_overrides or {}).get(member.sid)
            member_pwd = (req.member_pwd_overrides or {}).get(member.sid)
            result = await _worker_workspace_preflight(
                host,
                member_path,
                member.on_node,
                replacements=replacements,
                pwd_override=member_pwd or req.pwd,
                require_ready=False,
            )
            results.append({"sid": member.sid, "on_node": member.on_node, **result})
        return {
            "legacy": False,
            "ready": all(item.get("ready", False) for item in results),
            "members": results,
            "gaps": [],
        }
    on_node = req.on_node if req is not None else "_host"
    if on_node != "_host":
        service = service_factory()
        host = getattr(service, "host", None)
        if host is None:
            raise HTTPException(status_code=503, detail="Lab host not configured")
        return await _worker_workspace_preflight(
            host,
            path,
            on_node,
            replacements=(req.workspace_overrides if req else None),
            pwd_override=(req.pwd if req else None),
            require_ready=False,
        )
    try:
        preflight = await asyncio.to_thread(preflight_session_workspaces, path)
    except WorkspaceResumeError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code.value, "message": str(exc)},
        ) from exc
    if preflight is None:
        meta = await asyncio.to_thread(read_session_meta, path)
        saved_pwd = meta.get("pwd")
        effective_pwd = (req.pwd if req is not None else None) or saved_pwd
        ready = bool(effective_pwd and Path(effective_pwd).is_dir())
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
                        "saved_pwd": saved_pwd,
                        "status": "invalid" if saved_pwd else "missing",
                        "creature_ids": [],
                    }
                ]
            ),
        }
    replacements = req.workspace_overrides if req is not None else None
    pwd_override = req.pwd if req is not None else None
    if pwd_override is not None:
        replacements = {
            member.creature_id: pwd_override for member in preflight.members
        }
    if replacements:
        raw_manifest = (await asyncio.to_thread(read_session_meta, path)).get(
            MANIFEST_KEY
        )
        assert isinstance(raw_manifest, dict)
        manifest = parse_manifest(raw_manifest)
        try:
            plan = plan_workspace_resume(
                manifest,
                replacements,
                allow_valid_targets=pwd_override is not None,
            )
        except WorkspaceResumeError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": exc.code.value, "message": str(exc)},
            ) from exc
        preflight = preflight_workspace_resume(plan.manifest)
    return {"legacy": False, **preflight_to_dict(preflight)}


class ResumeRequest(BaseModel):
    """Optional target, cluster membership, and working-directory overrides.

    An omitted body targets ``"_host"`` for compatibility. Cluster resumes
    require one ``(sid, on_node)`` pair per member so each worker adopts its
    own store before the host relinks them. Persisted ``cluster_members``
    metadata supplies the list when the caller does not.
    """

    on_node: str = "_host"
    members: list[ClusterMember] | None = None
    # Explicit compatibility override: broadcast one directory to the team.
    pwd: str | None = None
    # Preferred modern form: only named creatures/path groups are replaced.
    workspace_overrides: dict[str, str] | None = None
    # Cluster form: member session id -> that node's targeted replacements.
    member_workspace_overrides: dict[str, dict[str, str]] | None = None
    member_pwd_overrides: dict[str, str] | None = None


@router.post("/{session_name}/resume")
async def resume_session(
    session_name: str,
    request: Request,
    req: ResumeRequest | None = None,
    session_dir: Path = Depends(resolve_request_session_dir),
    service_factory: Callable[[], TerrariumService] = Depends(get_service_factory),
):
    """Resume a saved session locally or on connected worker nodes.

    Local workspace validation finishes before the lazy service factory is
    called, so invalid resumes cannot allocate an engine runtime.
    """
    if req is not None and req.pwd is not None and req.workspace_overrides:
        raise HTTPException(
            status_code=422,
            detail="pwd and workspace_overrides are mutually exclusive",
        )
    on_node = (req.on_node if req is not None else "_host") or "_host"
    if (
        on_node == "_host"
        and getattr(request.app.state, "lab_mode", "standalone") == "lab-host"
    ):
        raise HTTPException(
            status_code=400,
            detail="Laboratory hosts cannot resume agents on _host; choose a worker",
        )

    path = await asyncio.to_thread(resolve_session_path_in, session_name, session_dir)
    if path is None:
        raise HTTPException(
            status_code=404, detail=f"Session not found: {session_name}"
        )

    if on_node == "_host":
        try:
            await asyncio.to_thread(
                prepare_resume_workspace,
                path,
                pwd=req.pwd if req is not None else None,
                workspace_overrides=(
                    req.workspace_overrides if req is not None else None
                ),
            )
        except WorkspaceResumeError as exc:
            status = 409 if exc.code.value == "unresolved" else 422
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code.value, "message": str(exc)},
            ) from exc
        except SessionNotResumableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        service = service_factory()
        if getattr(service, "host", None) is not None or hasattr(
            service, "connected_nodes"
        ):
            raise HTTPException(
                status_code=400,
                detail="Laboratory hosts cannot resume agents on _host; choose a worker",
            )
        try:
            session = await studio_resume(
                service,
                path,
                pwd_override=req.pwd if req is not None else None,
                workspace_overrides=(
                    req.workspace_overrides if req is not None else None
                ),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        instance_type = "terrarium" if len(session.creatures) > 1 else "agent"
        return {
            "instance_id": session.session_id,
            "type": instance_type,
            "session_name": session.name,
            "session": asdict(session),
        }

    service = service_factory()
    # Remote adoption requires both the lab transport host and a currently
    # connected target node.
    host = getattr(service, "host", None)
    connected = (
        list(service.connected_nodes()) if hasattr(service, "connected_nodes") else []
    )
    if host is None or on_node not in connected:
        raise HTTPException(
            status_code=404,
            detail=f"on_node={on_node!r} is not a connected lab node",
        )

    # Persisted cluster membership prevents a multi-worker session from being
    # resumed as an isolated singleton when the caller omits ``members``.
    requested_members = req.members if (req is not None and req.members) else None
    saved_members = await asyncio.to_thread(_read_saved_cluster_members, path)
    cluster_members = requested_members or saved_members
    if cluster_members and len(cluster_members) > 1:
        # Validate all targets before mutating workers to avoid a partially
        # resumed cluster.
        missing = [m for m in cluster_members if m.on_node not in connected]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=(
                    "CF-6 cluster resume: not every member's worker is "
                    f"connected (missing: {[m.on_node for m in missing]!r}). "
                    "Reconnect every worker named in cluster_members and "
                    "retry."
                ),
            )
        return await _resume_cluster(
            service,
            request,
            host,
            cluster_members,
            on_node,
            session_name,
            primary_sid=normalize_session_stem(path),
            pwd_override=req.pwd if req is not None else None,
            workspace_overrides=(req.workspace_overrides if req is not None else None),
            member_workspace_overrides=(
                req.member_workspace_overrides if req is not None else None
            ),
            member_pwd_overrides=(
                req.member_pwd_overrides if req is not None else None
            ),
            session_dir=session_dir,
        )

    await _worker_workspace_preflight(
        host,
        path,
        on_node,
        replacements=(req.workspace_overrides if req is not None else None),
        pwd_override=req.pwd if req is not None else None,
    )
    sid, meta, worker_pwd_exists = await _push_and_resume_member(
        host=host,
        request=request,
        path=path,
        on_node=on_node,
        pwd_override=req.pwd if req is not None else None,
        workspace_overrides=(req.workspace_overrides if req is not None else None),
    )
    try:
        mirror_snapshot = await asyncio.to_thread(
            _persist_remote_workspace_meta, path, meta, on_node
        )
    except BaseException as exc:
        cleanup_failures = await _cleanup_remote_sessions(host, [(on_node, sid)])
        if cleanup_failures or isinstance(exc, RemoteMirrorRollbackError):
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "partial_dirty",
                    "message": str(exc),
                    "cleanup_failures": cleanup_failures,
                },
            ) from exc
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        return await _build_remote_response(
            service,
            sid=sid,
            meta=meta,
            on_node=on_node,
            path=path,
            session_name=session_name,
            worker_pwd_exists=worker_pwd_exists,
        )
    except BaseException as exc:
        try:
            await asyncio.to_thread(_rollback_remote_workspace_meta, mirror_snapshot)
            rollback_failed = False
        except BaseException:
            rollback_failed = True
        cleanup_failures = await _cleanup_remote_sessions(host, [(on_node, sid)])
        if rollback_failed or cleanup_failures:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "partial_dirty",
                    "message": str(exc),
                    "cleanup_failures": cleanup_failures,
                },
            ) from exc
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def _resume_cluster(
    service: TerrariumService,
    request: Request,
    host,
    members: list[ClusterMember],
    primary_on_node: str,
    session_name: str,
    *,
    primary_sid: str,
    pwd_override: str | None = None,
    workspace_overrides: dict[str, str] | None = None,
    member_workspace_overrides: dict[str, dict[str, str]] | None = None,
    member_pwd_overrides: dict[str, str] | None = None,
    session_dir: Path,
) -> dict:
    """Preflight all members, then adopt and reconnect the cluster."""
    paths: dict[str, Path] = {}
    for member in members:
        resolved = await asyncio.to_thread(
            resolve_session_path_in, member.sid, session_dir
        )
        if resolved is None:
            raise HTTPException(
                status_code=404, detail=f"Session {member.sid!r} not found"
            )
        paths[member.sid] = resolved

    primary_member = next((m for m in members if m.sid == primary_sid), members[0])
    ordered = [primary_member] + [m for m in members if m.sid != primary_member.sid]
    if workspace_overrides and len(ordered) > 1:
        raise HTTPException(
            status_code=422,
            detail="Cluster workspace overrides must be scoped by member session ID",
        )
    member_overrides: dict[str, dict[str, str] | None] = {}
    for member in ordered:
        replacements = (member_workspace_overrides or {}).get(member.sid)
        if len(ordered) == 1 and replacements is None:
            replacements = workspace_overrides
        member_overrides[member.sid] = replacements
        member_pwd = (member_pwd_overrides or {}).get(member.sid)
        await _worker_workspace_preflight(
            host,
            paths[member.sid],
            member.on_node,
            replacements=replacements,
            pwd_override=member_pwd or pwd_override,
        )

    resumed: dict[str, tuple[str, dict, str]] = {}
    adopted: list[tuple[str, str]] = []
    mirror_snapshots = []
    registered_creatures: dict[str, str] = {}
    resumed_roster: tuple[Any, ...] = ()
    linked_creatures: list[tuple[str, str, str]] = []
    try:
        for member in ordered:
            new_sid, new_meta, _pwd_exists = await _push_and_resume_member(
                host=host,
                request=request,
                path=paths[member.sid],
                on_node=member.on_node,
                pwd_override=(member_pwd_overrides or {}).get(member.sid)
                or pwd_override,
                workspace_overrides=member_overrides[member.sid],
            )
            adopted.append((member.on_node, new_sid))
            resumed[member.sid] = (new_sid, new_meta, member.on_node)

        durable_members = [
            {"sid": resumed[item.sid][0], "on_node": item.on_node} for item in ordered
        ]
        for member in ordered:
            _sid, new_meta, node = resumed[member.sid]
            snapshot = await asyncio.to_thread(
                _persist_remote_workspace_meta,
                paths[member.sid],
                new_meta,
                node,
                cluster_members=(
                    durable_members if member.sid == primary_member.sid else None
                ),
            )
            mirror_snapshots.append(snapshot)

        primary_new_sid, primary_meta, primary_node = resumed[primary_member.sid]
        registered_creatures, resumed_roster = await _register_cluster_members(
            service, resumed, paths
        )
        primary_creature = registered_creatures.get(primary_member.sid)
        if primary_creature is None:
            raise HTTPException(
                status_code=502, detail="Primary worker returned no creature"
            )
        for member in ordered[1:]:
            peer = registered_creatures.get(member.sid)
            if peer is None:
                raise HTTPException(
                    status_code=502,
                    detail=f"Worker {member.on_node!r} returned no creature",
                )
            connection = await service.connect(primary_creature, peer)
            linked_creatures.append((primary_creature, peer, str(connection.channel)))
    except BaseException as exc:
        disconnect_failures = []
        for sender, receiver, channel in reversed(linked_creatures):
            try:
                await service.disconnect(sender, receiver, channel=channel)
            except BaseException:
                disconnect_failures.append(f"{sender}:{receiver}:{channel}")
        _unregister_cluster_members(service, resumed, registered_creatures)
        rollback_failures = []
        for snapshot in reversed(mirror_snapshots):
            try:
                await asyncio.to_thread(_rollback_remote_workspace_meta, snapshot)
            except BaseException:
                rollback_failures.append(str(snapshot.path))
        cleanup_failures = await _cleanup_remote_sessions(host, adopted)
        if (
            disconnect_failures
            or rollback_failures
            or cleanup_failures
            or isinstance(exc, RemoteMirrorRollbackError)
        ):
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "partial_dirty",
                    "message": str(exc),
                    "disconnect_failures": disconnect_failures,
                    "rollback_failures": rollback_failures,
                    "cleanup_failures": cleanup_failures,
                },
            ) from exc
        if isinstance(exc, (HTTPException, asyncio.CancelledError)):
            raise
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _build_cluster_response(
        primary_new_sid=primary_new_sid,
        primary_meta=primary_meta,
        primary_node=primary_node,
        resumed_roster=resumed_roster,
        resumed=resumed,
        ordered=ordered,
        session_name=session_name,
        primary_on_node=primary_on_node,
    )


def _read_saved_cluster_members(path: Path) -> list[ClusterMember] | None:
    """Read valid persisted cluster membership from a saved store.

    Missing, malformed, or singleton membership returns ``None``. The blocking
    store access must run through :func:`asyncio.to_thread`.
    """
    if not path.exists():
        return None
    try:
        raw = read_session_meta(path).get("cluster_members")
    except Exception:
        return None
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    members: list[ClusterMember] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("sid")
        node = entry.get("on_node")
        if isinstance(sid, str) and sid and isinstance(node, str) and node:
            members.append(ClusterMember(sid=sid, on_node=node))
    if len(members) < 2:
        return None
    return members
