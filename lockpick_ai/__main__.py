if __package__:
    from .app import run
else:
    # Also support direct execution/double-click of this file on Windows.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lockpick_ai.app import run

raise SystemExit(run())
