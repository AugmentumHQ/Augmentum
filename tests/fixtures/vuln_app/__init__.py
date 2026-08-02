"""Deliberately vulnerable FastAPI fixture used by the bug_finder
pen-test integration tests. NOT for any other purpose.

Every endpoint here is intentionally broken in a specific, documented
way. The pen-test integration tests boot this app and verify that
the http_attack / authz_matrix_probe / etc. primitives catch each
class of vulnerability.

DO NOT use this app as a starting point for real code.
"""
