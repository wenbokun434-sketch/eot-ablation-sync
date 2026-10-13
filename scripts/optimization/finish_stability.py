"""Finite O11 batch: two workers, then regression, replay audit and one TODO tick.

No O12 execution. A failed worker stops refill; existing workers retain progress.
Restart only once previous workers have exited; their checkpoints are resumed.
"""
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

import psutil


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finalize(root):
    out = root / "outputs/optimization/v1/stability"
    audit = json.loads((out / "o11_acceptance.json").read_text(encoding="utf-8"))
    preflight = json.loads((out / "reports/o11/preflight.json").read_text(encoding="utf-8"))
    if (audit.get("task") != "O11" or audit.get("status") != "pass"
            or audit.get("new_runs") != preflight["expected_new_runs"]
            or audit.get("reused_runs") != len(preflight["reused_seed42_runs"])
            or audit.get("preflight_sha256") != sha(out / "reports/o11/preflight.json")):
        raise ValueError("O11 does not have a current full acceptance")
    if set(audit["report_sha256"]) != set(map(str, range(5))):
        raise ValueError("Five outer reports are required")
    for f, expected in audit["report_sha256"].items():
        if sha(out / f"reports/outer_{f}.json") != expected:
            raise ValueError("O11 report changed after acceptance")
    for name, expected in preflight["frozen_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Frozen sources changed after acceptance")
    path = root / "优化TODO.md"
    original = path.read_text(encoding="utf-8")
    done = re.findall(r"^- \[x\] \*\*(O\d{2})", original, re.M)
    if done == [f"O{i:02}" for i in range(1, 12)]:
        return
    if done != [f"O{i:02}" for i in range(1, 11)]:
        raise ValueError("O11 must be the only next task")
    updated, count = re.subn(r"^- \[ \] \*\*O11", "- [x] **O11", original, count=1, flags=re.M)
    if count != 1:
        raise ValueError("Missing unchecked O11")
    updated += f"""

### O11 完成验收（{datetime.now(timezone.utc).isoformat()}）

- 修改目标：补齐 A 和 O10 晋级候选的种子 43/44，形成三种子稳定性验证。
- 允许修改的范围：本 TODO、新增 O11 编排/验收/测试与 stability 独立产物。
- 不允许破坏的逻辑：冻结训练实现、配置、模型及旧产物不变；仅 H_j 参与选择；不挑最好种子；不读取外层测试成绩；不执行 O12。
- 验收标准：所需三种子完整，先平均概率后重选双阈值，逐折/逐种子差值完整，固定稳定性判定可重放，保护哈希一致。
- 实际修改：新增 O11 训练编排、预测来源核验、三种子集成、稳定性报告和有限批次收尾入口；复用 {audit['reused_runs']} 次种子 42 运行，完成 {audit['new_runs']} 次新增运行。
- 验收命令与结果：优化及全部回归通过；python tests/optimization/audit_o11.py 返回 pass，逐折重放预测/双阈值/指标及稳定性判定，保护文件 {audit['protected_files']} 项一致。
- 证据路径：outputs/optimization/v1/stability/o11_acceptance.json、reports/outer_0.json 至 outer_4.json、reports/o11/ 下的测试日志和控制器状态。
- 任务状态：验收通过；仅勾选 O11。内层稳定性结果不代表最终测试改善。
- 下一任务：O12，按报告条件检查组合或 Focal；本控制器不执行该任务。
"""
    temporary = path.with_suffix(".md.o11.tmp")
    temporary.write_text(updated, encoding="utf-8")
    if path.read_text(encoding="utf-8") != original:
        raise RuntimeError("TODO changed concurrently")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / "outputs/optimization/v1/stability/reports/o11"
    directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (directory / "controller.lock").open("a+b")
    lock.seek(0, os.SEEK_END)
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    for proc in psutil.process_iter(["pid", "cmdline", "cwd"]):
        info = proc.info
        if (info["cwd"] and Path(info["cwd"]).resolve() == root
                and any(x.replace("\\", "/").endswith("scripts/optimization/stability.py")
                        for x in info["cmdline"] or [])):
            raise RuntimeError(f"Existing O11 worker {info['pid']}; do not launch duplicates")
    preflight = json.loads((directory / "preflight.json").read_text(encoding="utf-8"))
    state = {"task": "O11", "pid": os.getpid(), "expected_new_runs": preflight["expected_new_runs"]}
    def save(**updates):
        state.update(updates, updated_at_utc=datetime.now(timezone.utc).isoformat())
        temp = directory / "controller.json.tmp"
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp.replace(directory / "controller.json")
        print(json.dumps(updates), flush=True)
    def unchanged():
        for name, expected in preflight["frozen_sha256"].items():
            if sha(root / name) != expected:
                raise RuntimeError(f"Source/O10 evidence changed: {name}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    def launch(command, name):
        with (directory / name).open("ab") as log:
            return subprocess.Popen([sys.executable, *command], cwd=root, env=env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    def checked(command, name):
        unchanged()
        if launch(command, name).wait() != 0:
            raise RuntimeError(f"O11 acceptance command failed: {name}")
    running = {}
    try:
        unchanged()
        # Reports are immutable and were written only after complete training.
        pending = [f for f in range(5) if not (directory.parent / f"outer_{f}.json").exists()]
        completed = [f for f in range(5) if f not in pending]
        while pending or running:
            finished = []
            for f, proc in running.items():
                code = proc.poll()
                if code is not None:
                    if code != 0 or not (directory.parent / f"outer_{f}.json").exists():
                        raise RuntimeError(f"O11 outer {f} failed with exit {code}; stop refill")
                    finished.append(f)
            for f in finished:
                del running[f]
                completed.append(f)
            if pending and len(running) < 2:
                unchanged()
                while pending and len(running) < 2:
                    f = pending.pop(0)
                    running[f] = launch(["scripts/optimization/stability.py", "--root", str(root),
                        "--outer-fold", str(f), "--device", "cuda"], f"outer_{f}.log")
            save(status="training", worker_pids={str(f): p.pid for f, p in running.items()},
                 completed_folds=sorted(completed), pending_folds=pending)
            if running:
                time.sleep(15)
        save(status="auditing", worker_pids={})
        # Fresh logs on retries; preserve failed evidence under a unique suffix.
        for name in ("optimization_tests.log", "original_tests.log", "audit.log"):
            path = directory / name
            if path.exists():
                path.rename(directory / f"{name}.before_{os.getpid()}")
        checked(["-m", "unittest", "discover", "-s", "tests/optimization", "-v"], "optimization_tests.log")
        checked(["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], "original_tests.log")
        checked(["tests/optimization/audit_o11.py"], "audit.log")
        unchanged()
        finalize(root)
        save(status="complete", next_task="O12 (not started)")
    except Exception as exc:
        save(status="failed", error=f"{type(exc).__name__}: {exc}",
             worker_pids={str(f): p.pid for f, p in running.items() if p.poll() is None}, todo_checked=False)
        raise


if __name__ == "__main__":
    main()
