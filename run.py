"""
GraphLens Launcher
Single command to start the dashboard.
"""
import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("  GraphLens - GraphRAG Inference Hackathon")
    print("  By Pratham Bandre & Vinit Prajapati")
    print("=" * 60)
    print()
    print("  Starting Streamlit Dashboard...")
    print("  Dashboard: http://localhost:8501")
    print("  Mode: MOCK (Demo) - No API keys required")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.port", "8501",
    ])

if __name__ == "__main__":
    main()
