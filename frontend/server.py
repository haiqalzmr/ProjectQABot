"""
Flask web server for the Policy Q&A Bot.
Provides API endpoints and serves the web interface.
"""

from flask import Flask, render_template, request, jsonify


def create_app(pipeline):
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    @app.route("/")
    def index():
        """Serve the main chat interface."""
        stats = pipeline.get_stats()
        return render_template("index.html", stats=stats)

    @app.route("/api/ask", methods=["POST"])
    def ask():
        """
        Q&A endpoint.
        Request:  {"question": "..."}
        Response: {"answer": "...", "citations": "...", "sources": [...], "confidence": 0.xx}
        """
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field."}), 400

        question = data["question"].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty."}), 400

        try:
            result = pipeline.ask(question)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Pipeline error: {str(e)}"}), 500

    @app.route("/api/stats", methods=["GET"])
    def stats():
        """Return pipeline statistics."""
        try:
            return jsonify(pipeline.get_stats())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app
