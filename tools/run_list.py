"""Run a list of commands in parallel, one core each; works on Windows, macOS and Linux (no xargs needed).

    python tools/run_list.py runs/r4_6.txt            # as many at once as there are cores, minus one
    python tools/run_list.py runs/r4_6.txt -j 6       # 6 at once
    python tools/run_list.py runs/r4_6.txt --dry      # show what would run

Each line of the run list is written in shell style, e.g.
    OMP_NUM_THREADS=1 python bench.py --model sparc ... > logs/x.log 2>&1
Leading VAR=value assignments become environment variables, `> file` (and `2>&1`) send output to that log file,
and `python` is replaced by the interpreter running this script. Every job also gets OMP/OPENBLAS/MKL threads = 1
(one core per run; many runs in parallel is faster than one multi-threaded run for SPARC).
A failed job is retried once; a summary is printed at the end.
"""
import argparse, os, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONE_THREAD = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1'}


def parse(line):
    toks = shlex.split(line, posix=True)
    env, i = {}, 0
    while i < len(toks) and '=' in toks[i] and not toks[i].startswith('-') and toks[i].split('=', 1)[0].isidentifier():
        k, v = toks[i].split('=', 1); env[k] = v; i += 1
    cmd, log = [], None
    toks = toks[i:]
    j = 0
    while j < len(toks):
        t = toks[j]
        if t in ('>', '1>'):
            log = toks[j + 1]; j += 2; continue
        if t.startswith('>') and len(t) > 1 and t != '>&':
            log = t[1:]; j += 1; continue
        if t in ('2>&1',):
            j += 1; continue
        cmd.append(t); j += 1
    if cmd and cmd[0] in ('python', 'python3'):
        cmd[0] = sys.executable
    return env, cmd, log


def run(job, retries=1):
    n, line = job
    env_add, cmd, log = parse(line)
    env = {**os.environ, **ONE_THREAD, **env_add}
    for attempt in range(retries + 1):
        t0 = time.time()
        if log:
            os.makedirs(os.path.dirname(os.path.join(ROOT, log)) or '.', exist_ok=True)
            with open(os.path.join(ROOT, log), 'w') as f:
                rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        else:
            rc = subprocess.call(cmd, cwd=ROOT, env=env)
        dt = time.time() - t0
        if rc == 0:
            print(f'[{n}] ok   {dt:7.0f}s  {log or " ".join(cmd)}', flush=True)
            return n, 0
        print(f'[{n}] FAIL rc={rc} (attempt {attempt + 1})  {log or " ".join(cmd)}', flush=True)
    return n, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runlist')
    ap.add_argument('-j', '--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.runlist, encoding='utf-8') if l.strip() and not l.strip().startswith('#')]
    jobs = list(enumerate(lines, 1))
    if a.dry:
        for n, l in jobs:
            env, cmd, log = parse(l)
            print(n, env, cmd, '->', log)
        return
    print(f'{len(jobs)} jobs, {a.jobs} at a time', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run, jobs))
    bad = [n for n, rc in res if rc]
    print(f'done in {time.time() - t0:.0f}s; {len(jobs) - len(bad)} ok, {len(bad)} failed' + (f': {bad}' if bad else ''))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
