"""Start the interactive explorer: the transmission chains of the project, step by step, in a browser.

    python scripts/explore.py                  # opens http://127.0.0.1:8077/
    python scripts/explore.py --port 9000 --no-browser
    python scripts/explore.py --device cpu     # force the learned models onto the CPU

Every stage shown in the page is computed by the project's own code (semcom.channel, semcom.digital,
semcom.models, semcom.metrics). The trained models in results/models/ are needed: run scripts/run_all.py first.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from semcom import chdir_project_root  # noqa: E402

chdir_project_root()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--device", help="cpu ou cuda (défaut : cuda si disponible)")
    a = ap.parse_args()
    if a.device:
        os.environ["SEMCOM_DEVICE"] = a.device

    from explorer.catalog import SYSTEMS, trained  # noqa: E402
    from explorer.server import serve  # noqa: E402

    missing = sorted({n for s in SYSTEMS for n in s["needs"] if not trained(n)})
    if missing:
        print("modèles manquants (les systèmes concernés seront grisés) :", ", ".join(missing))
        print("  entraînez-les avec : python scripts/run_all.py")
    serve(a.host, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
