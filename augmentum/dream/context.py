"""Dream context window retrieval and memory clustering."""
from __future__ import annotations

from datetime import datetime, timezone

from augmentum.dream.models import ContextSegment
from augmentum.state.tree_utils import linearize_to_node, find_node_by_evidence


MAX_WINDOW_MESSAGES = 12  # 6 pairs cap


class DreamContextBuilder:

    def extract_window(
        self,
        path: list[dict],
        target_id: str,
        pairs: int = 3,
    ) -> list[dict]:
        """Extract message window centered on target node from linearized path.
        Returns up to `pairs * 2` messages centered on the target position.
        """
        pos = next((i for i, m in enumerate(path) if m["id"] == target_id), None)
        if pos is None:
            return []
        window_size = pairs * 2
        half = window_size // 2
        start = max(0, pos - half)
        end = start + window_size
        if end > len(path):
            end = len(path)
            start = max(0, end - window_size)
        window = path[start:end]
        return window[:MAX_WINDOW_MESSAGES]

    def cluster_by_proximity(
        self,
        memories: list[dict],
        window_size: int = 3,
    ) -> list[dict]:
        """Group memories by session and proximity.
        Memories within overlapping windows in the same session are merged.
        Returns list of cluster dicts with 'memories' and 'window_size' keys.
        """
        by_session: dict[str, list[dict]] = {}
        for mem in memories:
            sid = mem["session_id"]
            by_session.setdefault(sid, []).append(mem)

        clusters = []
        for session_id, session_mems in by_session.items():
            # Sort key needs to be comparable — `dict.get(k, default)` only
            # returns the default when the key is ABSENT, not when its value
            # is None. The memories SELECT always includes source_message_id
            # as a column, so the key is present even when NULL in the DB,
            # and `.get("source_message_id", "")` returns None rather than
            # "". Sorting None against None then raises TypeError on Py3.
            # Coerce explicitly so the key is always a comparable scalar.
            def _sort_key(m: dict):
                pos = m.get("position")
                if pos is not None:
                    return (0, pos)  # numeric path — keep ordering distinct from string path
                return (1, m.get("source_message_id") or "")
            session_mems.sort(key=_sort_key)
            current_cluster = [session_mems[0]]
            for mem in session_mems[1:]:
                if self._are_proximate(current_cluster[-1], mem, window_size):
                    current_cluster.append(mem)
                else:
                    clusters.append(self._make_cluster(current_cluster, session_id))
                    current_cluster = [mem]
            clusters.append(self._make_cluster(current_cluster, session_id))
        return clusters

    def _are_proximate(self, a: dict, b: dict, window_size: int) -> bool:
        """Check if two memories are close enough to merge windows.
        Uses 'position' field if available (resolved by engine), otherwise falls
        back to numeric extraction from source_message_id (for testing).
        """
        pos_a = a.get("position")
        pos_b = b.get("position")
        if pos_a is not None and pos_b is not None:
            return abs(pos_b - pos_a) <= window_size * 2
        # Fallback: extract digits from source_message_id (test fixtures use m1, m2, etc.)
        try:
            num_a = int("".join(c for c in a.get("source_message_id", "") if c.isdigit()) or "0")
            num_b = int("".join(c for c in b.get("source_message_id", "") if c.isdigit()) or "0")
            return abs(num_b - num_a) <= window_size * 2
        except (ValueError, TypeError):
            return False

    def _make_cluster(self, memories: list[dict], session_id: str) -> dict:
        return {
            "memories": memories,
            "session_id": session_id,
            "window_size": min(len(memories) * 6, MAX_WINDOW_MESSAGES),
        }

    def humanize_age(self, timestamp: str, now: datetime | None = None) -> str:
        """Convert absolute timestamp to relative age string."""
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            delta = now - dt
            if delta.days == 0:
                hours = delta.seconds // 3600
                if hours == 0:
                    return "just now"
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            if delta.days == 1:
                return "yesterday"
            if delta.days < 7:
                return f"{delta.days} days ago"
            if delta.days < 30:
                weeks = delta.days // 7
                return f"{weeks} week{'s' if weeks != 1 else ''} ago"
            months = delta.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        except (ValueError, TypeError):
            return "some time ago"
