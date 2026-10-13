"""Finish this O10 batch, adopting existing workers; never starts O11.

This is a finite job controller, not a scheduler. Failed subprocesses stop the
batch. TODO is checked only after the final manifest and independent audit pass.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

from eot.optimization.config import file_sha256


def rolling_schedule(adopted, completed, launch_fold, poll, verify_report, save,
                     unchanged, sleep=time.sleep):
    """Refill two slots in fold order, only after checking every active worker."""
    if len(adopted) > 2 or set(adopted) - set(range(5)):
        raise ValueError("At most two valid outer folds may be adopted")
    running = dict(adopted)
    pending = [f for f in range(5) if f not in running and f not in completed]
    while running or pending:
        unchanged()
        finished = []
        # Check all exits before launching anything: a failure must stop refill.
        for fold, proc in running.items():
            done, code = poll(proc)
            if done:
                if code not in (None, 0):
                    raise RuntimeError(f"Outer {fold} failed with exit {code}; stop refill")
                verify_report(fold)
                finished.append(fold)
        for fold in finished:
            del running[fold]
            completed.add(fold)
        while pending and len(running) < 2:
            fold = pending.pop(0)
            running[fold] = launch_fold(fold)
        save(status="training", scheduling="rolling_two_workers",
             active_folds=sorted(running), pending_folds=list(pending),
             completed_folds=sorted(completed),
             worker_pids={str(f): p.pid for f, p in running.items()})
        if running:
            sleep(5)


def finalize_todo(root, audit):
    screen = root / "outputs/optimization/v1/screening"
    if (audit.get("task") != "O10" or audit.get("status") != "pass" or audit.get("runs") != 60
            or file_sha256(screen / "screening_manifest.json") != audit.get("screening_manifest_sha256")):
        raise ValueError("Final audit is not a current, passing 60-run O10 acceptance")
    for relative, sha in audit["artifact_sha256"].items():
        if file_sha256(root / relative) != sha:
            raise ValueError("Source changed after acceptance")
    path = root / "优化TODO.md"
    original = path.read_text(encoding="utf-8")
    completed = re.findall(r"^- \[x\] \*\*(O\d{2})", original, re.M)
    if completed == [f"O{i:02}" for i in range(1, 11)]:
        return
    if completed != [f"O{i:02}" for i in range(1, 10)]:
        raise ValueError("TODO order changed; do not update task state automatically")
    updated, count = re.subn(r"^- \[ \] \*\*O10", "- [x] **O10", original, count=1, flags=re.M)
    if count != 1:
        raise ValueError("O10 is not the single next unchecked task")
    updated += f"""

### O10 完成验收（后台收尾，{datetime.now(timezone.utc).isoformat()}）

- 修改目标：完成种子 42 的 A–D、五个外层、三个内层的 60 次初筛及独立验收。
- 允许修改的范围：O10 必要编排修复、测试、本 TODO 与 screening 独立产物。
- 不允许破坏的逻辑：原训练配置/源码/数据保持冻结；H_j 只用于内层选择；不读外层测试成绩；不进入 O11。
- 验收标准：60 次运行、预测、双阈值与逐外层比较完整，测试及独立审计通过，保护哈希一致。
- 实际修改：接续已启动进程，按外层编号滚动调度、最多并发两个训练进程；只复用或恢复原运行；补齐 manifest 和独立审计。
- 验收命令与结果：screening manifest 为 pass；优化测试、原测试和 `python tests/optimization/audit_o10.py` 全部退出码为 0；审计确认 60 次运行与 {audit['protected_files']} 个保护文件。详见 `outputs/optimization/v1/screening/o10_acceptance.json`。
- 证据路径：`outputs/optimization/v1/screening/reports/o10/controller.json`、`controller.log`、测试日志及 `o10_acceptance.json`。上方进行中记录为历史快照，以本节最终验收为准。
- 任务状态：验收通过；仅勾选 O10。内层筛选成绩不代表独立最终测试性能。
- 下一任务：O11，按本次逐外层晋级结果补齐基线和候选的种子 43/44；本控制器不执行该任务。
"""
    temporary = path.with_suffix(".md.o10.tmp")
    temporary.write_text(updated, encoding="utf-8")
    if path.read_text(encoding="utf-8") != original:
        raise RuntimeError("TODO changed concurrently; preserve the user's edit")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--adopt", nargs="*", default=[], help="outer-fold:existing-PID")
    parser.add_argument("--replace-watcher", type=int)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    directory = root / "outputs/optimization/v1/screening/reports/o10"
    directory.mkdir(parents=True, exist_ok=True)
    # Windows releases this lock on controller exit; workers are not terminated.
    import msvcrt
    controller_lock = (directory / "controller.lock").open("a+b")
    controller_lock.seek(0, os.SEEK_END)
    if controller_lock.tell() == 0:
        controller_lock.write(b"0")
        controller_lock.flush()
    controller_lock.seek(0)
    msvcrt.locking(controller_lock.fileno(), msvcrt.LK_NBLCK, 1)
    sources = [root / p for p in ("src/eot/optimization/screening.py", "scripts/optimization/screening.py",
        "tests/optimization/test_screening.py", "tests/optimization/test_screening_controller.py",
        "tests/optimization/audit_o10.py", "scripts/optimization/finish_screening.py")]
    # Keep the waiting controller light: hash sources without importing Torch.
    sources.extend((root / "src/eot").glob("*.py"))
    sources.extend((root / "src/eot/optimization").glob("*.py"))
    expected = {str(p): file_sha256(p) for p in sources}
    state = {"task": "O10", "status": "starting", "pid": os.getpid(), "source_sha256": expected}
    def save(**updates):
        state.update(updates, updated_at_utc=datetime.now(timezone.utc).isoformat())
        temp = directory / "controller.json.tmp"
        temp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(directory / "controller.json")
        print(json.dumps(updates), flush=True)
    def unchanged():
        if any(file_sha256(Path(p)) != sha for p, sha in expected.items()):
            raise RuntimeError("O10 source changed while controller waited; stop before further work")
    adopted = {}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    def launch(command, name):
        unchanged()
        log = (directory / name).open("ab")
        try:
            return subprocess.Popen([sys.executable, *command], cwd=root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        finally:
            log.close()
    def checked(command, name):
        proc = launch(command, name)
        if proc.wait() != 0:
            raise RuntimeError(f"Command failed: {name}; see retained log")
    def fold_command(fold):
        return ["scripts/optimization/screening.py", "--root", str(root), "--device", "cuda",
                "run-outer", "--outer-fold", str(fold)]
    try:
        for item in args.adopt:
            fold, pid = map(int, item.split(":"))
            proc = psutil.Process(pid)
            command = proc.cmdline()
            if (fold not in range(5) or fold in adopted or Path(proc.cwd()).resolve() != root
                    or not any(c.replace("\\", "/").endswith("scripts/optimization/screening.py") for c in command)
                    or command[-2:] != ["--outer-fold", str(fold)]):
                raise ValueError("Adoption must identify the exact current screening worker")
            adopted[fold] = proc
        # Every existing training worker must be accounted for before refill.
        live_workers = set()
        for proc in psutil.process_iter(["pid", "cmdline", "cwd"]):
            try:
                info = proc.info
                if (info["cwd"] and Path(info["cwd"]).resolve() == root
                        and any(c.replace("\\", "/").endswith("scripts/optimization/screening.py")
                                for c in info["cmdline"] or [])
                        and "run-outer" in (info["cmdline"] or [])):
                    live_workers.add(info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if live_workers != {p.pid for p in adopted.values()}:
            raise ValueError("Existing screening workers do not match adoption; refusing duplicate launch")
        if args.replace_watcher:
            watcher = psutil.Process(args.replace_watcher)
            if not any(c.lower().endswith("o10_watcher.ps1") for c in watcher.cmdline()):
                raise ValueError("Refusing to stop an unrelated watcher")
            watcher.terminate()  # Only the dispatcher; adopted training processes continue.
            watcher.wait(timeout=10)
        save(status="waiting_existing", adopted={str(f): p.pid for f, p in adopted.items()})
        # Fresh files prevent an old passing log from hiding a later failure.
        for name in ("optimization_tests.log", "original_tests.log", "audit.log"):
            target = directory / name
            if target.exists():
                target.rename(directory / (name + f".before_{os.getpid()}"))
        def verify_report(fold):
            path = root / f"outputs/optimization/v1/screening/reports/outer_{fold}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("task") != "O10" or report.get("outer_fold") != fold:
                raise ValueError(f"Invalid screening report for outer {fold}")
        def poll(proc):
            if isinstance(proc, psutil.Process):
                try:
                    return True, proc.wait(timeout=0)
                except psutil.TimeoutExpired:
                    return False, None
            code = proc.poll()
            return code is not None, code
        completed = set()
        for fold in range(5):
            if fold not in adopted and (root / f"outputs/optimization/v1/screening/reports/outer_{fold}.json").exists():
                verify_report(fold)
                completed.add(fold)
        rolling_schedule(adopted, completed,
                         lambda f: launch(fold_command(f), f"outer_{f}.log"),
                         poll, verify_report, save, unchanged)
        save(status="auditing", active_folds=[])
        # Run CPU regression suites after GPU workers exit to avoid extra memory
        # pressure on the 16 GB host. This changes no training parameter/budget.
        checked(["-m", "unittest", "discover", "-s", "tests/optimization", "-v"], "optimization_tests.log")
        checked(["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], "original_tests.log")
        save(tests="pass")
        checked(["scripts/optimization/screening.py", "--root", str(root), "manifest"], "manifest.log")
        checked(["tests/optimization/audit_o10.py"], "audit.log")
        audit = json.loads((root / "outputs/optimization/v1/screening/o10_acceptance.json").read_text(encoding="utf-8"))
        unchanged()
        finalize_todo(root, audit)
        save(status="complete", next_task="O11 (not started)")
    except Exception as exc:
        save(status="failed", error=f"{type(exc).__name__}: {exc}", todo_checked=False)
        raise


if __name__ == "__main__":
    main()
