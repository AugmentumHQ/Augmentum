"""Tests for GitHub README link rewriting in the browse reader."""

from augmentum.proxy.browse_routes import _rewrite_github_relative_links


def test_github_root_relative_repo_link_targets_github_site_root():
    html = '<a href="/anthropics/claude-code">Claude Code</a>'

    out = _rewrite_github_relative_links(html, "curator", "awesome-list", "main")

    assert 'href="https://github.com/anthropics/claude-code"' in out
    assert "curator/awesome-list/blob/main/anthropics/claude-code" not in out


def test_github_relative_file_link_targets_repo_blob():
    html = '<a href="./docs/guide.md">Guide</a>'

    out = _rewrite_github_relative_links(html, "curator", "awesome-list", "main")

    assert 'href="https://github.com/curator/awesome-list/blob/main/docs/guide.md"' in out
