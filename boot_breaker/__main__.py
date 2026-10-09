if __package__:
    from .app import run
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from boot_breaker.app import run

raise SystemExit(run())
