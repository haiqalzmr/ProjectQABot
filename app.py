"""
Policy Q&A Bot — Entry Point
Supports CLI mode and web server mode.

Usage:
    python app.py --web              # Launch web UI (default)
    python app.py --question "..."   # Ask a single question
    python app.py --interactive      # Interactive CLI mode
    python app.py --rebuild          # Force rebuild the index
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          # Suppress TF info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'         # Suppress oneDNN messages

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Policy Q&A Bot with Grounded Citations"
    )
    parser.add_argument(
        "--web", action="store_true", default=True,
        help="Launch the web interface (default)"
    )
    parser.add_argument(
        "--question", "-q", type=str, default=None,
        help="Ask a single question from the command line"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive CLI mode"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild the FAISS index"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for the web server (default: 5000)"
    )

    args = parser.parse_args()

    # Import pipeline
    from core.pipeline import PolicyQAPipeline
    from config import WEB_HOST, WEB_PORT

    # Initialize pipeline
    pipeline = PolicyQAPipeline()
    pipeline.initialize(force_rebuild=args.rebuild)

    # Single question mode
    if args.question:
        result = pipeline.ask(args.question)
        print(f"\nQ: {result['question']}")
        print(f"\n{result['answer']}")
        if result['citations']:
            print(f"\n{result['citations']}")
        return

    # Interactive CLI mode
    if args.interactive:
        print("\n=== Interactive Mode (type 'quit' to exit) ===\n")
        while True:
            try:
                question = input("You: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not question:
                    continue

                result = pipeline.ask(question)
                print(f"\nAssistant: {result['answer']}")
                if result['citations']:
                    print(f"\n{result['citations']}")
                print()
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
        return

    # Web mode (default)
    from frontend.server import create_app

    port = args.port or WEB_PORT
    app = create_app(pipeline)
    print(f"\n🌐 Starting web server at http://{WEB_HOST}:{port}")
    print("   Press Ctrl+C to stop.\n")
    app.run(host=WEB_HOST, port=port, debug=False)


if __name__ == "__main__":
    main()
