"""Branch tracker — DAG-based message tracking with branch detection.

Detects when a frontend replays a conversation from an earlier point
(e.g., SillyTavern swipe/regenerate, Augmentum edit/sibling swap) and
manages state accordingly.

Detection strategy:
1. Hash each message content
2. On each request, compare the message sequence to known history
3. If the sequence diverges from the known history, detect a branch point
4. Fork state at the branch point

Branch IDs are content-based (hash of the divergent message sequence)
so swapping back to the same path reuses the same branch ID, enabling
state restoration.
"""

from __future__ import annotations

import hashlib

from augmentum.models.base import InternalChatRequest
from augmentum.state.narrative_state import TrackedMessage, content_hash
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class BranchDetection:
    """Result of branch detection analysis."""

    def __init__(
        self,
        *,
        is_branch: bool = False,
        branch_point: int = -1,
        new_branch_id: str = "",
        parent_branch_id: str = "main",
    ) -> None:
        self.is_branch = is_branch
        self.branch_point = branch_point
        self.new_branch_id = new_branch_id
        self.parent_branch_id = parent_branch_id


class BranchTracker:
    """Tracks messages as a DAG and detects branching."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._messages: list[TrackedMessage] = []
        self._current_branch = "main"
        self._branch_counter = 0

    @property
    def messages(self) -> list[TrackedMessage]:
        return list(self._messages)

    @property
    def current_branch(self) -> str:
        return self._current_branch

    @property
    def active_messages(self) -> list[TrackedMessage]:
        """Get messages in the current active branch."""
        return [m for m in self._messages if m.branch_id == self._current_branch and m.is_active]

    def detect_branch(self, request: InternalChatRequest) -> BranchDetection:
        """Analyze a request to detect if it represents a branch from history.

        Compares the message sequence in the request against our tracked history.
        If the sequence diverges, we've detected a branch.
        """
        # Build hash sequence from request messages (skip system messages)
        request_hashes = []
        for msg in request.messages:
            if msg.role != "system":
                request_hashes.append((msg.role, content_hash(msg.content)))

        # Build hash sequence from tracked history
        history_hashes = []
        active = self.active_messages
        for msg in active:
            if msg.role != "system":
                history_hashes.append((msg.role, msg.content_hash))

        if not history_hashes:
            # No history yet — not a branch
            return BranchDetection()

        # Find the divergence point
        divergence = -1
        min_len = min(len(request_hashes), len(history_hashes))

        for i in range(min_len):
            if request_hashes[i] != history_hashes[i]:
                divergence = i
                break

        # Case 1: Request is shorter than history — user went back
        if divergence == -1 and len(request_hashes) < len(history_hashes):
            # The request matches up to its length, but history continues.
            # Two distinct user actions can produce this shape:
            #
            #   (a) REGEN of the last response — UI drops just the last
            #       assistant message and resends. shortage == 1. Engine's
            #       legacy regen path (engine.py:281+) handles this:
            #       replace-in-place, no branch, no rollback. Preserve.
            #
            #   (b) REWIND to an earlier point — user clicked back to
            #       message 3 of a 25-message conversation. shortage > 1.
            #       Memory tiers (STATE/LEDGER/ARCHIVE) need to roll back
            #       to that point or the inspector keeps showing message-25
            #       state. We fire is_branch=True so:
            #         - rollback_to(branch_point) prunes ledger past cutoff
            #         - snapshot-history prefetch restores STATE-at-cutoff
            #         - retrieval bounds archive to the new ancestry
            #
            # The shortage threshold (>1) cleanly separates these: regenerate
            # always drops exactly the trailing assistant; deliberate rewind
            # drops the trailing pair plus everything beyond.
            #
            # branch_id is content-addressed on the truncated prefix so going
            # back to the same point twice rejoins the same branch (idempotent).
            shortage = len(history_hashes) - len(request_hashes)
            if shortage <= 1:
                return BranchDetection()  # legacy regen path

            branch_point = len(request_hashes)
            branch_fingerprint = hashlib.sha256(
                ("rewind:" + str(request_hashes)).encode()
            ).hexdigest()[:12]
            new_branch_id = f"branch_{branch_fingerprint}"
            log.info(
                "branch_detected_rewind",
                divergence_point=branch_point,
                new_branch=new_branch_id,
                parent_branch=self._current_branch,
                history_len=len(history_hashes),
                request_len=len(request_hashes),
                shortage=shortage,
            )
            return BranchDetection(
                is_branch=True,
                branch_point=branch_point,
                new_branch_id=new_branch_id,
                parent_branch_id=self._current_branch,
            )

        # Case 2: Request extends history — normal continuation
        if divergence == -1 and len(request_hashes) >= len(history_hashes):
            return BranchDetection()

        # Case 3: Divergence detected — this is a branch.
        # Generate a content-based branch ID from the divergent message
        # sequence so the same path always gets the same ID (enables
        # state save/restore when users swap between branches).
        divergent_hashes = request_hashes[divergence:]
        branch_fingerprint = hashlib.sha256(
            str(divergent_hashes).encode()
        ).hexdigest()[:12]
        new_branch_id = f"branch_{branch_fingerprint}"

        log.info(
            "branch_detected",
            divergence_point=divergence,
            new_branch=new_branch_id,
            parent_branch=self._current_branch,
            history_len=len(history_hashes),
            request_len=len(request_hashes),
        )

        return BranchDetection(
            is_branch=True,
            branch_point=divergence,
            new_branch_id=new_branch_id,
            parent_branch_id=self._current_branch,
        )

    def apply_branch(self, detection: BranchDetection) -> None:
        """Apply a detected branch — soft-delete divergent messages and switch branch."""
        if not detection.is_branch:
            return

        # Soft-delete messages after the branch point in the current branch
        for msg in self._messages:
            if (
                msg.branch_id == detection.parent_branch_id
                and msg.message_index >= detection.branch_point
                and msg.role != "system"
            ):
                msg.is_active = False

        self._current_branch = detection.new_branch_id

        log.info(
            "branch_applied",
            new_branch=detection.new_branch_id,
            branch_point=detection.branch_point,
        )

    def track_message(
        self,
        role: str,
        content: str,
        message_index: int | None = None,
        parent_id: int | None = None,
    ) -> TrackedMessage:
        """Track a new message in the DAG."""
        if message_index is None:
            # Auto-assign based on active messages in current branch
            active = self.active_messages
            non_system = [m for m in active if m.role != "system"]
            message_index = len(non_system)

        msg = TrackedMessage(
            session_id=self._session_id,
            parent_id=parent_id,
            role=role,
            content=content,
            content_hash=content_hash(content),
            message_index=message_index,
            branch_id=self._current_branch,
            is_active=True,
        )
        self._messages.append(msg)
        return msg

    def track_request_messages(self, request: InternalChatRequest) -> list[TrackedMessage]:
        """Track new messages from a request, position-aware.

        Walks the request's non-system messages in order and compares each
        to the tracked message at the same logical position on the current
        active branch. A match at position i means this message is already
        tracked — skip. Otherwise it's new — track it.

        The previous implementation deduped purely by content_hash anywhere
        in history (and ignored role). Two messages with identical content
        at different positions — common in long chats with repeated short
        replies like "continue" or shared greetings — silently collapsed
        into one tracked message, leaving the tracker N messages short of
        the request. ``detect_branch`` then flagged every subsequent turn
        as a branch because the index-by-index hash comparison was off by
        the dedup count, producing the false-branch behavior we were seeing.
        Comparing at the same position fixes the alignment.
        """
        tracked = []
        active_non_system = [m for m in self.active_messages if m.role != "system"]
        pos = 0
        for msg in request.messages:
            if msg.role == "system":
                continue
            already_tracked = (
                pos < len(active_non_system)
                and active_non_system[pos].content_hash == content_hash(msg.content)
                and active_non_system[pos].role == msg.role
            )
            if not already_tracked:
                tracked.append(self.track_message(msg.role, msg.content))
            pos += 1
        return tracked

    def get_branch_point_index(self, branch_id: str) -> int:
        """Get the message index where a branch diverged."""
        branch_messages = [m for m in self._messages if m.branch_id == branch_id]
        if branch_messages:
            return branch_messages[0].message_index
        return 0

    def set_messages(self, messages: list[TrackedMessage]) -> None:
        """Set messages directly (used when loading from DB)."""
        self._messages = messages
        # Determine current branch from the most recent active message
        active = [m for m in messages if m.is_active]
        if active:
            self._current_branch = active[-1].branch_id
