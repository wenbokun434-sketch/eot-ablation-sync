"""Finite O12 controller: at most two folds, then regression/audit, only O12 tick."""
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
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def finalize(root):
    out = root / 'outputs/optimization/v1/conditional'
    audit = json.loads((out / 'o12_acceptance.json').read_text(encoding='utf-8'))
    pre = json.loads((out / 'reports/o12/preflight.json').read_text(encoding='utf-8'))
    if (audit.get('task') != 'O12' or audit.get('status') != 'pass'
            or not pre['initial_run_count'] <= audit.get('conditional_runs', -1) <= pre['maximum_conditional_runs']
            or audit.get('preflight_sha256') != sha(out / 'reports/o12/preflight.json')
            or set(audit['report_sha256']) != set(map(str, range(5)))):
        raise ValueError('Current complete O12 acceptance is required')
    for f, expected in audit['report_sha256'].items():
        if sha(out / f'reports/outer_{f}.json') != expected:
            raise ValueError('O12 report changed after acceptance')
    for name, expected in pre['frozen_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen sources changed after acceptance')
    path = root / '优化TODO.md'
    original = path.read_text(encoding='utf-8')
    done = re.findall(r'^- \[x\] \*\*(O\d{2})', original, re.M)
    if done == [f'O{i:02}' for i in range(1, 13)]:
        return
    if done != [f'O{i:02}' for i in range(1, 12)]:
        raise ValueError('O12 must be the only next task')
    updated, count = re.subn(r'^- \[ \] \*\*O12', '- [x] **O12', original, count=1, flags=re.M)
    if count != 1:
        raise ValueError('Missing unchecked O12')
    updated += f'''

### O12 完成验收（{datetime.now(timezone.utc).isoformat()}）

- 修改目标：按冻结条件执行组合或最终分支 Focal 实验。
- 允许修改的范围：新增 O12 配置/业务模块/CLI/测试、conditional 独立产物与本 TODO。
- 不允许破坏的逻辑：原数据/网络/训练参数及旧产物不变；前缀仍 BCE；仅 H_j 参与判断；不读取外层测试成绩，不进入 O13。
- 验收标准：数值与梯度检查通过，触发运行完整、三种子集成阈值重选及条件结果可重放，保护哈希一致，回归及独立审计通过后仅勾选 O12。
- 实际修改：新增 Focal 最终分支与独立配置/训练适配器，原 O07 训练函数体在独立命名空间中接入损失构建器；完成 {audit['conditional_runs']} 次条件训练，所有内层唯一运行总数 {audit['total_unique_inner_runs']}，未超过 270 次预算。
- 验收命令与结果：优化与全量回归通过；python tests/optimization/audit_o12.py 返回 pass；{audit['protected_files']} 个保护文件一致，各外层触发/不触发及稳定性判定可由保存概率重放。
- 证据路径：outputs/optimization/v1/conditional/o12_acceptance.json、reports/outer_0.json 至 outer_4.json、reports/o12/ 日志及各候选报告。
- 任务状态：验收通过；仅勾选 O12。条件实验内层结果不代表外层最终性能改善。
- 下一任务：O13；本控制器及本次授权均不执行 O13。
'''
    temporary = path.with_suffix('.md.o12.tmp')
    temporary.write_text(updated, encoding='utf-8')
    if path.read_text(encoding='utf-8') != original:
        raise RuntimeError('TODO changed concurrently')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    directory = root / 'outputs/optimization/v1/conditional/reports/o12'
    directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (directory / 'controller.lock').open('a+b')
    lock.seek(0, os.SEEK_END)
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    for proc in psutil.process_iter(['pid', 'cmdline', 'cwd']):
        info = proc.info
        if (info['cwd'] and Path(info['cwd']).resolve() == root
                and any(x.replace('\\', '/').endswith('scripts/optimization/conditional.py')
                        for x in info['cmdline'] or [])):
            raise RuntimeError(f"Existing O12 worker {info['pid']}; refuse duplicate launch")
    pre = json.loads((directory / 'preflight.json').read_text(encoding='utf-8'))
    state = {'task': 'O12', 'pid': os.getpid(), 'initial_runs': pre['initial_run_count'],
             'maximum_runs': pre['maximum_conditional_runs']}
    def save(**updates):
        state.update(updates, updated_at_utc=datetime.now(timezone.utc).isoformat())
        temporary = directory / 'controller.json.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(directory / 'controller.json')
        print(json.dumps(updates), flush=True)
    def unchanged():
        for name, expected in pre['frozen_sha256'].items():
            if sha(root / name) != expected:
                raise RuntimeError(f'O12 source/evidence changed: {name}')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    def launch(command, name):
        with (directory / name).open('ab') as log:
            return subprocess.Popen([sys.executable, *command], cwd=root, env=env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    def checked(command, name):
        unchanged()
        if launch(command, name).wait() != 0:
            raise RuntimeError(f'O12 acceptance command failed: {name}')
    running = {}
    try:
        unchanged()
        pending = [f for f in range(5) if not (directory.parent / f'outer_{f}.json').exists()]
        completed = [f for f in range(5) if f not in pending]
        while pending or running:
            finished = []
            for f, proc in running.items():
                code = proc.poll()
                if code is not None:
                    if code != 0 or not (directory.parent / f'outer_{f}.json').exists():
                        raise RuntimeError(f'O12 outer {f} failed with exit {code}; stop refill')
                    finished.append(f)
            for f in finished:
                del running[f]
                completed.append(f)
            if pending and len(running) < 2:
                unchanged()
                while pending and len(running) < 2:
                    f = pending.pop(0)
                    running[f] = launch(['scripts/optimization/conditional.py', '--root', str(root),
                        '--outer-fold', str(f), '--device', 'cuda'], f'outer_{f}.log')
            save(status='training', worker_pids={str(f):p.pid for f,p in running.items()},
                 completed_folds=sorted(completed), pending_folds=pending)
            if running:
                time.sleep(15)
        save(status='auditing', worker_pids={})
        for name in ('optimization_tests.log', 'original_tests.log', 'audit.log'):
            path = directory / name
            if path.exists():
                path.rename(directory / f'{name}.before_{os.getpid()}')
        checked(['-m', 'unittest', 'discover', '-s', 'tests/optimization', '-v'], 'optimization_tests.log')
        checked(['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'], 'original_tests.log')
        checked(['tests/optimization/audit_o12.py'], 'audit.log')
        unchanged()
        finalize(root)
        save(status='complete', next_task='O13 (not started)')
    except Exception as exc:
        save(status='failed', error=f'{type(exc).__name__}: {exc}',
             worker_pids={str(f):p.pid for f,p in running.items() if p.poll() is None}, todo_checked=False)
        raise


if __name__ == '__main__':
    main()
