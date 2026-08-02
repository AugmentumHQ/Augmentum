"""Media server integrations — Audiobookshelf / Emby / Jellyfin.

Each user configures their own instances; catalog items land in the
unified `file_index` so the existing file browser handles search,
filtering, progress, and trash without special-casing media rows.
Streams are proxied through ``/api/media/stream/{file_id}`` so user
tokens never leave the server and CORS / mixed-content combinations
work uniformly across deployments.
"""
