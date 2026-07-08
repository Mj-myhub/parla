"""Hugging Face Spaces entrypoint. Makes src/ importable, then runs the real app."""
import sys, pathlib, runpy
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
runpy.run_path(str(pathlib.Path(__file__).parent / "src/parla/app/streamlit_app.py"),
               run_name="__main__")
