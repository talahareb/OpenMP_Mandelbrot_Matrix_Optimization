#!/usr/bin/env python3
"""Collect real timings to docs/benchmarks/*.csv (no simulated data)."""
import argparse
import csv
from pathlib import Path
import platform
import re
import subprocess
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks"

def run(*args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mandelbrot-width', type=int, default=1300)
    parser.add_argument('--mandelbrot-height', type=int, default=900)
    parser.add_argument('--matrix-sizes', type=int, nargs='+', default=[128,256,384])
    parser.add_argument('--threads', type=int, nargs='+', default=[1,2,4])
    parser.add_argument('--repeats', type=int, default=3)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    mrows=[]
    for mode, thread, sched in [('serial',1,'static')] + [('omp', t, s) for t in args.threads for s in ('static','dynamic')]:
        for trial in range(1,args.repeats+1):
            stdout=run('./mandelbrot/mandelbrot','--mode',mode,'--threads',str(thread),'--schedule',sched,
                '--width',str(args.mandelbrot_width),'--height',str(args.mandelbrot_height),
                '--max-iter','255','--no-image')
            record=dict(re.findall(r'(\w+)=([^\s]+)',stdout))
            mrows.append(dict(mode=mode,threads=thread,schedule=sched,trial=trial,
                              width=args.mandelbrot_width,height=args.mandelbrot_height,
                              time_s=float(record['time_s']),pixel_hash=record['hash']))
    with (OUT/'mandelbrot_timings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mrows[0]));w.writeheader();w.writerows(mrows)
    assert len({r['pixel_hash'] for r in mrows})==1,'Different image hashes: invalid results'
    matrows=[]
    for size in args.matrix_sizes:
        for t in args.threads:
            stdout=run('./matrix/matmul','--size',str(size),'--threads',str(t),'--repeats',str(args.repeats))
            matrows += list(csv.DictReader(stdout.splitlines()))
    assert all(r['correct']=='1' for r in matrows),'Invalid matrix output'
    with (OUT/'matrix_timings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(matrows[0]));w.writeheader();w.writerows(matrows)
    info={'platform':platform.platform(),'python':platform.python_version(),
          'processor':platform.processor(),'note':'One machine, short measurements; run again for system-specific results'}
    cpu=run('bash','-lc','lscpu | grep -E "Model name:|CPU\\(s\\):" | head -3 || true')
    cc=run('gcc','--version').splitlines()[0]
    (OUT/'benchmark_environment.txt').write_text('\n'.join(f'{k}: {v}' for k,v in info.items())+'\n'+cpu+'\n'+cc+'\n')
    print(f"Saved {len(mrows)} Mandelbrot observations and {len(matrows)} matrix observations.")
    def mmed(rs):return median([float(r['time_s']) for r in rs])
    for t in args.threads:
        for s in ('static','dynamic'):
            print(f"Mandelbrot omp {t} {s}: {mmed([r for r in mrows if r['mode']=='omp' and r['threads']==t and r['schedule']==s]):.5f}s")
    print('Mandelbrot serial: {:.5f}s'.format(mmed([r for r in mrows if r['mode']=='serial'])))

if __name__=='__main__':main()
