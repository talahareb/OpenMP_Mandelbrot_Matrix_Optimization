#!/usr/bin/env python3
"""Rebuild portfolio visuals directly from computed images and benchmark CSVs."""
from pathlib import Path
import csv
import math
import subprocess
import tempfile
from statistics import median
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs'/'images'
BENCH=ROOT/'docs'/'benchmarks'
OUT.mkdir(exist_ok=True,parents=True)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def capture_mandelbrot(filename,kwargs):
    with tempfile.TemporaryDirectory() as temp:
        pgm=Path(temp)/'out.pgm'
        cmd=[str(ROOT/'mandelbrot'/'mandelbrot'),'--mode','omp','--threads','4',
             '--output',str(pgm),*kwargs]
        info=subprocess.run(cmd,cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()
        im=Image.open(pgm).convert('L')
        im.save(OUT/filename,optimize=True)
        return im,info

print('Generating actual fractal renders...')
classic,info=capture_mandelbrot('mandelbrot-grayscale.png',
    ['--width','1280','--height','880','--max-iter','255'])
zoom,zoom_info=capture_mandelbrot('mandelbrot-detail-grayscale.png',
    ['--width','1280','--height','880','--max-iter','750',
     '--center-real','-0.7435','--center-imag','0.1314','--span','0.015'])

# Color map is an optional presentation layer; the underlying pixel values
# are from the C renderer. No image generator or fabricated pixels.
def colorize(im):
    a=np.asarray(im,dtype=np.uint8)
    z=a.astype(np.float32)/255.0
    # Deep ink interior and electric-teal/amber contours.
    red=np.clip(14+240*np.power(z,1.7),0,255).astype(np.uint8)
    green=np.clip(33+210*np.sin(z*math.pi*0.79)**2,0,255).astype(np.uint8)
    blue=np.clip(54+195*np.sqrt(np.minimum(z,1)),0,255).astype(np.uint8)
    rgb=np.stack([red,green,blue],axis=-1)
    rgb[a==255]=[12,25,40]
    return Image.fromarray(rgb,'RGB')

colorize(classic).save(OUT/'mandelbrot-color.png',optimize=True)
colorize(zoom).save(OUT/'mandelbrot-detail.png',optimize=True)

# Two actual computed images featured in a compact, readable banner.
w=1400;h=470
canvas=Image.new('RGB',(w,h),'#0d1827')
d=ImageDraw.Draw(canvas)
large=ImageFont.truetype(FONT_BOLD,34)
small=ImageFont.truetype(FONT,19)
d.text((35,22),'MANDELBROT SET',font=large,fill='#e9f2fd')
d.text((37,70),'Sequential and OpenMP rendering · row-wise parallelism',font=small,fill='#a6c1d4')
for im,x,label in [(colorize(classic),30,'FULL VIEW'),(colorize(zoom),710,'ZOOM DETAIL')]:
    panel=im.copy()
    panel.thumbnail((660,335),Image.Resampling.LANCZOS)
    px=x+(660-panel.width)//2
    py=105+(330-panel.height)//2
    canvas.paste(panel,(px,py))
    d.text((x+10,443),label,font=ImageFont.truetype(FONT_BOLD,15),fill='#b0ccd9')
canvas.save(OUT/'mandelbrot-showcase.png',optimize=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':15,
                     'axes.titleweight':'bold','axes.labelcolor':'#24364b',
                     'text.color':'#24364b','axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#d5dfe8','grid.color':'#e4eaf0',
                     'figure.facecolor':'white'})

def get_csv(file):
    with (BENCH/file).open() as f:return list(csv.DictReader(f))

def med(rows):return median([float(r['time_s']) for r in rows])
mand=get_csv('mandelbrot_timings.csv')
serial=med([r for r in mand if r['mode']=='serial'])
threads=sorted({int(r['threads']) for r in mand if r['mode']=='omp'})
fig,ax=plt.subplots(figsize=(10,4.8),dpi=175)
for sched,color,marker in [('static','#146b85','o'),('dynamic','#e58f41','s')]:
    ratios=[]
    for t in threads:
        rs=[r for r in mand if r['mode']=='omp' and r['schedule']==sched and int(r['threads'])==t]
        ratios.append(serial/med(rs))
    ax.plot(threads,ratios,marker=marker,linewidth=2.3,markersize=8,color=color,label=f'OpenMP · {sched}')
ax.plot(threads,threads,linestyle='--',color='#94a3b8',linewidth=1.2,label='Ideal linear speedup')
ax.set_xticks(threads)
ax.set_xlabel('Number of CPU threads')
ax.set_ylabel('Speedup vs. sequential (×)')
ax.set_title('Mandelbrot rendering | measured CPU scaling',loc='left',pad=18)
ax.grid(axis='y',alpha=.65)
ax.legend(frameon=False,ncol=3,loc='upper left',fontsize=9)
ax.text(0.99,-.22,'Median of recorded trials · wall-clock time · 1300 × 900 pixels',
        ha='right',transform=ax.transAxes,fontsize=8,color='#64748b')
fig.tight_layout(pad=2)
fig.savefig(OUT/'mandelbrot-speedup.png',bbox_inches='tight');plt.close(fig)

matrix=get_csv('matrix_timings.csv')
sizes=sorted({int(r['size']) for r in matrix})
variants=['naive','openmp','transpose','transpose-openmp']
labels=['Naive C','OpenMP','Transposed B','Transposed + OpenMP']
colors=['#94a3b8','#2687a1','#e5a056','#4554a4']
fig,ax=plt.subplots(figsize=(10.4,5.1),dpi=175)
barw=0.18
x=np.arange(len(sizes))
maxthreads=max(int(r['threads']) for r in matrix)
for i,(var,label,color) in enumerate(zip(variants,labels,colors)):
    vals=[]
    for n in sizes:
        rows=[r for r in matrix if int(r['size'])==n and r['variant']==var and int(r['threads'])==(maxthreads if 'openmp' in var else 1)]
        vals.append(med(rows)*1000)
    ax.bar(x+(i-1.5)*barw,vals,width=barw,label=label,color=color)
ax.set_xticks(x,sizes)
ax.set_xlabel('Square matrix dimension (N × N)')
ax.set_ylabel('Median computation time (ms) · lower is better')
ax.set_title(f'Matrix multiplication | four strategies compared',loc='left',pad=18)
ax.grid(axis='y',alpha=.6)
ax.set_axisbelow(True)
ax.legend(frameon=False,ncol=2,loc='upper left',fontsize=9)
ax.text(0.99,-.22,f'Transposition time included · OpenMP bars use {maxthreads} threads · 3 trials',
        ha='right',transform=ax.transAxes,fontsize=8,color='#64748b')
fig.tight_layout(pad=2)
fig.savefig(OUT/'matrix-comparison.png',bbox_inches='tight');plt.close(fig)

# Draw a conceptual access-pattern illustration (not a measurement).
fig,axs=plt.subplots(1,2,figsize=(11,4.8),dpi=150)
for ax,mode in zip(axs,['Original B: column access','Transposed Bᵀ: row access']):
    ax.set_title(mode,pad=13,fontsize=14,fontweight='bold')
    ax.set_aspect('equal')
    for r in range(5):
        for c in range(5):
            hit=(c==2 if mode.startswith('Original') else r==2)
            ax.add_patch(plt.Rectangle((c,4-r),.94,.94,facecolor=('#1b779a' if hit else '#edf2f8'),edgecolor='white',lw=1.5))
            ax.text(c+.47,4-r+.47,f'{r},{c}',ha='center',va='center',fontsize=10,color=('white' if hit else '#547184'))
    ax.set_xlim(-.1,5.1);ax.set_ylim(-.1,5.1);ax.axis('off')
fig.suptitle('Why transpose? Contiguous memory access for dot products',fontsize=16,fontweight='bold',y=1.04)
fig.tight_layout(pad=1)
fig.savefig(OUT/'matrix-memory-access.png',bbox_inches='tight');plt.close(fig)
print('Rendered:', ', '.join(p.name for p in sorted(OUT.glob('*.png'))))
print('Fractal run:',info)
print('Zoom run:',zoom_info)
