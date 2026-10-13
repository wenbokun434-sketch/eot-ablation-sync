"""Finite O14 controller: >=1 / <=N final-training workers, C calibration, one OOF test.

No test label is touched until all five post-calibration locks verify. A failed
worker stops refill and leaves existing workers/checkpoints in place. Snapshots
are appended every 20 minutes for external supervision; only a current O14 audit
allows the single TODO O14 tick.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

try:
    import psutil
except Exception:  # pragma: no cover - only used for duplicate-process detection
    psutil = None

FINAL_SEEDS = (42, 43, 44)


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def preflight(root: Path) -> dict:
    from eot.optimization.config import load_optimization_config
    from eot.optimization.final import FINAL_SOURCES, prepare_final_run

    directory = root / "outputs/optimization/v1/final/reports/o14"
    directory.mkdir(parents=True, exist_ok=True)
    audit = subprocess.run(
        [sys.executable, "outputs/optimization/v1/locked/audit_o13.py", "verify"],
        cwd=root, capture_output=True, text=True, check=False)
    if audit.returncode != 0:
        raise RuntimeError(f"O13 verify failed: {audit.stderr.strip()}")
    config = load_optimization_config("configs/optimization/A.yaml", project_root=root)
    expected = []
    run_directories = set()
    for outer_fold in range(5):
        for seed in FINAL_SEEDS:
            plan = prepare_final_run(config, outer_fold=outer_fold, seed=seed)
            expected.append({"outer_fold": outer_fold, "seed": seed,
                             "run_id": plan.identity.run_id,
                             "run_directory": plan.identity.directory.relative_to(root).as_posix(),
                             "epochs": plan.o13_record["epochs"]})
            run_directories.add(plan.identity.directory.relative_to(root).as_posix())
    if len(expected) != 15 or len(run_directories) != 15:
        raise RuntimeError("O14 preflight must yield 15 unique final run identities")
    frozen = {name: sha(root / name) for name in (FINAL_SOURCES +
        ("outputs/optimization/v1/locked/audit_o13.py",
         "outputs/optimization/v1/locked/final_lock.json",
         "outputs/optimization/v1/protocol/baseline_protection.json",
         "configs/optimization/A.yaml", "scripts/optimization/finish_final.py"))}
    preflight_record = {
        "task": "O14",
        "status": "pass",
        "expected_runs": 15,
        "seeds": list(FINAL_SEEDS),
        "outer_folds": 5,
        "expected": expected,
        "config_sha256": config.config_hash,
        "protocol_sha256": config.protocol_sha256,
        "o13_lock_sha256": "43782083a4a847bc594ef998e65d969f2d651cac439f54e559afdc11f477a711",
        "audit_command": "python outputs/optimization/v1/locked/audit_o13.py verify",
        "audit_stdout": audit.stdout.strip(),
        "frozen_sha256": frozen,
        "test_access": "closed_until_all_five_post_calibration_locks_verified",
    }
    _write_atomic(directory / "preflight.json",
                  json.dumps(preflight_record, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return preflight_record


def _tail(path: Path, lines: int = 2) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return content[-lines:]


def finalize(root: Path) -> None:
    acceptance_path = root / "outputs/optimization/v1/final/o14_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    preflight_path = root / "outputs/optimization/v1/final/reports/o14/preflight.json"
    preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (acceptance.get("task") != "O14" or acceptance.get("status") != "pass"
            or acceptance.get("final_runs") != 15 or acceptance.get("second_locks") != 5
            or acceptance.get("oof_users") != 909
            or acceptance.get("evaluation_summary_sha256") != sha(
                root / "outputs/optimization/v1/final/evaluation/o14_evaluation_summary.json")
            or acceptance.get("preflight_sha256") != sha(preflight_path)):
        raise ValueError("O14 audit is not a current complete acceptance")
    for name, expected in preflight_record["frozen_sha256"].items():
        if sha(root / name) != expected:
            raise RuntimeError(f"Frozen O14 source/evidence changed after preflight: {name}")

    path = root / "优化TODO.md"
    original = path.read_text(encoding="utf-8")
    done = re.findall(r"^- \[x\] \*\*(O\d{2})", original, re.M)
    if done == [f"O{i:02}" for i in range(1, 15)]:
        return
    if done != [f"O{i:02}" for i in range(1, 14)]:
        raise ValueError("O14 must be the only next task before ticking")
    updated, count = re.subn(r"^- \[ \] \*\*O14", "- [x] **O14", original, count=1, flags=re.M)
    if count != 1:
        raise ValueError("Missing unchecked O14")
    metrics = json.loads((root / "outputs/optimization/v1/final/evaluation/o14_evaluation_summary.json")
                         .read_text(encoding="utf-8"))
    r2 = metrics["objects"]["R2"]["counts"]
    opt = metrics["objects"]["OPT"]["counts"]
    updated += f'''

### O14 完成验收（{datetime.now(timezone.utc).isoformat()}）

- 修改目标：按 O13 锁完成五折 × 三种子最终重训、C 校准与唯一一次外层评估。
- 允许修改的范围：本文件、O14 最终训练/校准/预测/指标独立目录，以及最小新增 O14 适配/CLI/审计。
- 不允许破坏的逻辑：C/T 不参与候选与 epoch 选择；S 重新拟合统计；不迁移内层阈值/checkpoint；五锁齐备后才访问外层测试；原保护文件不改写。
- 验收标准：15 次最终训练与五折第二份锁完整；909 名 OOF 用户唯一；双阈值与固定回退可重放；全量回归与 O14 独立审计通过；仅勾选 O14。
- 实际修改：新增 O14 固定 epoch 最终训练、C 阈值校准、五折密封锁、唯一外层评估、有限并发控制器与独立审计；全部外层均按 O13 锁定回退 A/R2。
- 验收命令与结果：`python tests/optimization/audit_o14.py` 返回 pass；{acceptance['protected_files']} 个保护文件一致；OOF 用户 {acceptance['oof_users']}；R2 TP/TN/FP/FN={r2['tp']}/{r2['tn']}/{r2['fp']}/{r2['fn']}，OPT TP/TN/FP/FN={opt['tp']}/{opt['tn']}/{opt['fp']}/{opt['fn']}；流程双指标变化按 evaluation summary 如实报告。
- 证据路径：outputs/optimization/v1/final/o14_acceptance.json、locks/outer_0.json 至 outer_4.json、evaluation/o14_evaluation_summary.json、reports/o14/ 日志与快照。
- 任务状态：验收通过；本轮仅勾选 O14。内层无稳定候选，OPT 在全部外层固定回退为 A/R2；是否相对历史 R0 双提升见冻结评估报告，不虚报改善。
- 下一任务：O15，交付优化报告、Notebook 与最终审计；本控制器不执行 O15。
'''
    temporary = path.with_suffix(".md.o14.tmp")
    temporary.write_text(updated, encoding="utf-8")
    if path.read_text(encoding="utf-8") != original:
        raise RuntimeError("TODO changed concurrently")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--snapshot-seconds", type=int, default=20 * 60)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("O14 controller supports 1 or 2 workers on the locked 8GB profile")
    root = args.root.resolve()
    directory = root / "outputs/optimization/v1/final/reports/o14"
    directory.mkdir(parents=True, exist_ok=True)

    import msvcrt
    lock = (directory / "controller.lock").open("a+b")
    lock.seek(0, os.SEEK_END)
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if psutil is not None:
        for proc in psutil.process_iter(["pid", "cmdline", "cwd"]):
            info = proc.info
            if (info["cwd"] and Path(info["cwd"]).resolve() == root
                    and any(str(x).replace("\\", "/").endswith("scripts/optimization/finish_final.py")
                            for x in info["cmdline"] or [])):
                if proc.pid != os.getpid():
                    raise RuntimeError(f"Existing O14 controller {proc.pid}; refuse duplicate launch")

    preflight_record = preflight(root)
    state = {"task": "O14", "pid": os.getpid(), "workers": args.workers,
             "expected_runs": preflight_record["expected_runs"]}
    snapshots = directory / "supervision.jsonl"

    def save(**updates) -> None:
        state.update(updates, updated_at_utc=datetime.now(timezone.utc).isoformat())
        _write_atomic(directory / "controller.json",
                      json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        print(json.dumps(updates, ensure_ascii=False, sort_keys=True), flush=True)

    def snapshot(running: dict, completed: list, pending: list) -> None:
        record = {"task": "O14", "time_utc": datetime.now(timezone.utc).isoformat(),
                  "completed": [f"{f}_{s}" for f, s in sorted(completed)],
                  "pending": [f"{f}_{s}" for f, s in sorted(pending)],
                  "running": {f"{f}_{s}": {"pid": proc.pid, "tail": _tail(directory / f"outer_{f}_seed_{s}.log")}
                              for (f, s), proc in sorted(running.items())}}
        with snapshots.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def launch(command: list[str], name: str):
        with (directory / name).open("ab") as log:
            return subprocess.Popen([sys.executable, *command], cwd=root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=flags)

    def checked(command: list[str], name: str) -> None:
        if launch(command, name).wait() != 0:
            raise RuntimeError(f"O14 acceptance command failed: {name}")

    running: dict[tuple[int, int], subprocess.Popen] = {}
    completed: list[tuple[int, int]] = []
    pending = [(outer_fold, seed) for outer_fold in range(5) for seed in FINAL_SEEDS]
    failed = False
    last_snapshot = time.monotonic()
    try:
        save(status="training", completed=[], pending=[f"{f}_{s}" for f, s in pending])
        while pending or running:
            finished = []
            for key, proc in list(running.items()):
                code = proc.poll()
                if code is not None:
                    if code != 0:
                        failed = True
                        save(status="failed_worker", worker_pid=proc.pid,
                             failed_job=f"{key[0]}_{key[1]}", todo_checked=False)
                    else:
                        finished.append(key)
            for key in finished:
                del running[key]
                completed.append(key)
            if failed:
                if not running:
                    raise RuntimeError("O14 worker failed; stop refill and preserve checkpoints")
            elif pending and len(running) < args.workers:
                while pending and len(running) < args.workers:
                    outer_fold, seed = pending.pop(0)
                    name = f"outer_{outer_fold}_seed_{seed}.log"
                    running[(outer_fold, seed)] = launch([
                        "scripts/optimization/final.py", "train",
                        "--outer-fold", str(outer_fold), "--seed", str(seed),
                        "--device", args.device, "--resume"], name)
            save(status="training",
                 completed=[f"{f}_{s}" for f, s in sorted(completed)],
                 pending=[f"{f}_{s}" for f, s in sorted(pending)],
                 running={f"{f}_{s}": proc.pid for (f, s), proc in sorted(running.items())})
            if time.monotonic() - last_snapshot >= args.snapshot_seconds:
                snapshot(running, completed, pending)
                last_snapshot = time.monotonic()
            if running:
                time.sleep(args.poll_seconds)

        if len(completed) != 15:
            raise RuntimeError("O14 training ended without 15 complete runs")
        snapshot({}, completed, [])
        save(status="calibrating", completed=[f"{f}_{s}" for f, s in sorted(completed)], pending=[])
        for outer_fold in range(5):
            checked(["scripts/optimization/final.py", "calibrate",
                     "--outer-fold", str(outer_fold), "--device", args.device],
                    f"calibrate_outer_{outer_fold}.log")
        checked(["scripts/optimization/final.py", "verify-gate"], "verify_gate.log")
        save(status="evaluating")
        checked(["scripts/optimization/final.py", "evaluate", "--device", args.device],
                "evaluate.log")
        save(status="auditing")
        for name in ("optimization_tests.log", "original_tests.log", "audit.log"):
            path = directory / name
            if path.exists():
                path.rename(directory / f"{name}.before_{os.getpid()}")
        checked(["-m", "unittest", "discover", "-s", "tests/optimization", "-v"],
                "optimization_tests.log")
        checked(["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
                "original_tests.log")
        checked(["tests/optimization/audit_o14.py"], "audit.log")
        finalize(root)
        save(status="complete", next_task="O15 (not started)")
    except Exception as exc:
        save(status="failed", error=f"{type(exc).__name__}: {exc}", todo_checked=False,
             worker_pids={f"{f}_{s}": proc.pid for (f, s), proc in running.items()
                          if proc.poll() is None})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
