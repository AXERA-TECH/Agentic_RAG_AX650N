"""Scoped on-disk storage for chat attachments.

Attachments are addressed by opaque IDs.  In particular, chat clients can no
longer submit a local filesystem path: the only way bytes enter a model prompt
is through :meth:`AttachmentStore.save_bytes`.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agentic_rag.data.models import MediaType
from agentic_rag.services.media.models import StoredAttachment


class AttachmentError(ValueError):
    """Base error for attachment validation and lookup failures."""


class AttachmentNotFound(AttachmentError):
    """The requested attachment does not exist or belongs to another scope."""


class AttachmentTooLarge(AttachmentError):
    """The upload exceeds the configured size limit."""


class UnsupportedAttachment(AttachmentError):
    """The uploaded MIME type / extension is not a supported media kind."""


_MIME_TO_KIND = {
    "image/jpeg": MediaType.IMAGE,
    "image/jpg": MediaType.IMAGE,
    "image/png": MediaType.IMAGE,
    "image/webp": MediaType.IMAGE,
    "image/gif": MediaType.IMAGE,
    "image/bmp": MediaType.IMAGE,
    "audio/mpeg": MediaType.AUDIO,
    "audio/mp3": MediaType.AUDIO,
    "audio/wav": MediaType.AUDIO,
    "audio/x-wav": MediaType.AUDIO,
    "audio/mp4": MediaType.AUDIO,
    "audio/x-m4a": MediaType.AUDIO,
    "audio/ogg": MediaType.AUDIO,
    "audio/flac": MediaType.AUDIO,
    "audio/webm": MediaType.AUDIO,
    "video/mp4": MediaType.VIDEO,
    "video/quicktime": MediaType.VIDEO,
    "video/x-msvideo": MediaType.VIDEO,
    "video/webm": MediaType.VIDEO,
    "video/x-matroska": MediaType.VIDEO,
}

_EXT_TO_KIND = {
    ".jpg": MediaType.IMAGE,
    ".jpeg": MediaType.IMAGE,
    ".png": MediaType.IMAGE,
    ".webp": MediaType.IMAGE,
    ".gif": MediaType.IMAGE,
    ".bmp": MediaType.IMAGE,
    ".mp3": MediaType.AUDIO,
    ".wav": MediaType.AUDIO,
    ".m4a": MediaType.AUDIO,
    ".ogg": MediaType.AUDIO,
    ".flac": MediaType.AUDIO,
    ".mp4": MediaType.VIDEO,
    ".mov": MediaType.VIDEO,
    ".avi": MediaType.VIDEO,
    ".webm": MediaType.VIDEO,
    ".mkv": MediaType.VIDEO,
}

_MIME_BY_EXT = {
    ext: mime
    for ext, kind in _EXT_TO_KIND.items()
    for mime, mapped in _MIME_TO_KIND.items()
    if mapped == kind and mimetypes.guess_extension(mime) == ext
}

# A small fallback for extensions whose canonical guess differs between
# platforms (for example .jpg vs .jpe).
_MIME_BY_EXT.update({
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
})

_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_UNSAFE_NAME_RE = re.compile(r"[^0-9A-Za-z._\-一-鿿 ]+")

# WebM and Ogg are container formats used for both audio and video.  A
# browser recording is often ``audio/webm`` with a ``.webm`` extension; the
# extension must not force it into the video pipeline.
_AMBIGUOUS_CONTAINERS = {".webm", ".ogg"}


class AttachmentStore:
    """Store attachments under ``workspace/attachments/<id>/``."""

    def __init__(self, root: str | Path | None = None, max_bytes: int | None = None):
        if root is None or max_bytes is None:
            from agentic_rag.config.settings import get_settings
            settings = get_settings()
            if root is None:
                root = Path(settings.workspace_dir) / "attachments"
            if max_bytes is None:
                max_bytes = settings.media.max_attachment_bytes
        self.root = Path(root).resolve()
        self.max_bytes = int(max_bytes)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────

    def save_bytes(
        self,
        data: bytes,
        filename: str,
        mime: str = "",
        *,
        user_id: str = "",
        session_id: str = "",
    ) -> StoredAttachment:
        if not data:
            raise UnsupportedAttachment("Empty attachment")
        if len(data) > self.max_bytes:
            raise AttachmentTooLarge(
                f"Attachment exceeds the {self.max_bytes // (1024 * 1024)} MB limit"
            )

        safe_name = self._safe_filename(filename)
        kind, canonical_mime = self._detect_kind(mime, safe_name)
        attachment_id = uuid.uuid4().hex
        directory = self.root / attachment_id
        directory.mkdir(parents=True, exist_ok=False)

        suffix = Path(safe_name).suffix.lower()
        if not suffix:
            suffix = mimetypes.guess_extension(canonical_mime) or ".bin"
            safe_name = f"{safe_name}{suffix}"
        path = directory / safe_name
        try:
            path.write_bytes(data)
            attachment = StoredAttachment(
                id=attachment_id,
                kind=kind,
                path=str(path),
                filename=safe_name,
                mime=canonical_mime,
                size=len(data),
                user_id=user_id,
                session_id=session_id,
                created_at=_now(),
            )
            self._write_meta(attachment)
            return attachment
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    # ── Read ──────────────────────────────────────────────

    def get(
        self,
        attachment_id: str,
        *,
        user_id: str = "",
        session_id: str = "",
    ) -> StoredAttachment:
        if not _ID_RE.fullmatch(attachment_id or ""):
            raise AttachmentNotFound(f"Invalid attachment id: {attachment_id}")

        directory = (self.root / attachment_id).resolve()
        if not directory.is_relative_to(self.root) or not directory.is_dir():
            raise AttachmentNotFound(f"Attachment not found: {attachment_id}")

        meta = self._read_meta(directory)
        if meta is None:
            raise AttachmentNotFound(f"Attachment metadata missing: {attachment_id}")

        path = (directory / str(meta.get("filename") or "")).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise AttachmentNotFound(f"Attachment file missing: {attachment_id}")

        stored_user = str(meta.get("user_id") or "")
        stored_session = str(meta.get("session_id") or "")
        if user_id and stored_user and user_id != stored_user:
            raise AttachmentNotFound(f"Attachment not found: {attachment_id}")
        if session_id and stored_session and session_id != stored_session:
            raise AttachmentNotFound(f"Attachment not found: {attachment_id}")

        return StoredAttachment(
            id=attachment_id,
            kind=MediaType(meta.get("kind") or MediaType.IMAGE.value),
            path=str(path),
            filename=str(meta.get("filename") or path.name),
            mime=str(meta.get("mime") or "application/octet-stream"),
            size=int(meta.get("size") or 0),
            digest=str(meta.get("digest") or ""),
            user_id=stored_user,
            session_id=stored_session,
            created_at=str(meta.get("created_at") or ""),
        )

    def resolve(
        self,
        attachment_ids: Iterable[str],
        *,
        user_id: str = "",
        session_id: str = "",
    ) -> list[StoredAttachment]:
        resolved: list[StoredAttachment] = []
        seen: set[str] = set()
        for attachment_id in attachment_ids:
            if not attachment_id or attachment_id in seen:
                continue
            seen.add(attachment_id)
            resolved.append(self.get(
                attachment_id, user_id=user_id, session_id=session_id,
            ))
        return resolved

    def read_data_uri(self, attachment: StoredAttachment) -> str:
        data = Path(attachment.path).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{attachment.mime};base64,{encoded}"

    # ── Mutate / clean up ─────────────────────────────────

    def update_digest(self, attachment: StoredAttachment, digest: str) -> None:
        attachment.digest = digest
        self._write_meta(attachment)

    def delete(self, attachment_id: str) -> bool:
        if not _ID_RE.fullmatch(attachment_id or ""):
            return False
        directory = (self.root / attachment_id).resolve()
        if not directory.is_relative_to(self.root) or not directory.is_dir():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True

    # ── Internals ─────────────────────────────────────────

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path((filename or "").replace("\\", "/")).name.strip()
        name = _UNSAFE_NAME_RE.sub("_", name).strip(" .")
        return name or "attachment"

    @classmethod
    def _detect_kind(cls, mime: str, filename: str) -> tuple[MediaType, str]:
        mime = (mime or "").split(";", 1)[0].strip().lower()
        if mime == "application/octet-stream":
            mime = ""
        ext = Path(filename).suffix.lower()

        mime_kind = _MIME_TO_KIND.get(mime)
        ext_kind = _EXT_TO_KIND.get(ext)

        if mime_kind and ext_kind and mime_kind != ext_kind:
            ambiguous_av = (
                ext in _AMBIGUOUS_CONTAINERS
                and {mime_kind, ext_kind} <= {MediaType.AUDIO, MediaType.VIDEO}
            )
            if not ambiguous_av:
                raise UnsupportedAttachment(
                    f"MIME type {mime} does not match extension {ext}"
                )
        kind = mime_kind or ext_kind
        if kind is None:
            raise UnsupportedAttachment(
                f"Unsupported attachment type: mime={mime or 'unknown'}, ext={ext or 'none'}"
            )
        canonical_mime = mime if mime_kind else _MIME_BY_EXT.get(ext, "")
        return kind, canonical_mime or "application/octet-stream"

    def _write_meta(self, attachment: StoredAttachment) -> None:
        directory = Path(attachment.path).parent
        meta = {
            "id": attachment.id,
            "kind": attachment.kind.value,
            "filename": attachment.filename,
            "mime": attachment.mime,
            "size": attachment.size,
            "digest": attachment.digest,
            "user_id": attachment.user_id,
            "session_id": attachment.session_id,
            "created_at": attachment.created_at or _now(),
        }
        tmp = directory / "meta.json.tmp"
        tmp.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(directory / "meta.json")

    @staticmethod
    def _read_meta(directory: Path) -> dict | None:
        path = directory / "meta.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
