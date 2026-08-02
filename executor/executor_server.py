"""Sandboxed Python executor HTTP server."""

from __future__ import annotations

import traceback

from flask import Flask, jsonify, request
from sandbox import execute_code

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/execute", methods=["POST"])
def execute():
    try:
        data = request.get_json(force=True)
        code = data.get("code", "")
        timeout = data.get("timeout", 30)

        if not code.strip():
            return jsonify({"success": False, "error": "No code provided"}), 400

        result = execute_code(code, timeout=timeout)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
