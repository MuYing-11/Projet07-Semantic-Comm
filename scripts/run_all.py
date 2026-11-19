"""Train every model of the study in parallel (CPU), respecting dependencies (jscc_sem needs clsA).

    python scripts/run_all.py --workers 4 --threads 4
Finished models (results/models/<name>/info.json present) are skipped, so the script can be relaunched.
Logs: results/logs/<name>.log
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEPS = {"jscc_sem": ["clsA"]}
ORDER = ["clsA", "clsB", "jscc_mse", "dae", "jscc_sem", "jscc_mse_snr0", "jscc_mse_snr10", "jscc_mse_snr20", "task"]


def _job(name, threads):
    from semcom import chdir_project_root

    chdir_project_root()
    from semcom.train import train

    log = os.path.join("results", "logs", f"{name}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        sys.stdout = sys.stderr = fh
        try:
            train(name, threads=threads)
            return name, time.time() - t0, None
        except Exception:
            traceback.print_exc()
            return name, time.time() - t0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    from semcom.train import done

    todo = [n for n in (a.only or ORDER) if not done(n)]
    print("to train:", todo, flush=True)
    running = {}
    with ProcessPoolExecutor(a.workers) as ex:
        while todo or running:
            ready = [n for n in todo if all(done(d) for d in DEPS.get(n, []))]
            while ready and len(running) < a.workers:
                n = ready.pop(0)
                todo.remove(n)
                running[ex.submit(_job, n, a.threads)] = n
            if not running:
                print("blocked on dependencies:", todo)
                return
            fin, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in fin:
                name, dt, err = f.result()
                del running[f]
                print(f"{name}: {'FAILED' if err else 'done'} in {dt / 60:.1f} min", flush=True)
                if err:
                    print(err)


if __name__ == "__main__":
    main()
