#!/usr/bin/env python3
"""Regenerate independent fixtures: pip install mpmath (development only).

100-decimal arithmetic and LU solves; Rust tests need only the saved JSON.
No BoTorch cross-validation claim is made by these mathematical fixtures.
"""
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 100
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'crates/fcmaes-surrogate/tests/fixtures'
DEST.mkdir(parents=True, exist_ok=True)
zs = [-1000000, -1000, -40, -20, -10.000000001, -10, -8, -2, -1, 0, 1, 5, 10]
log_h = []
for z in zs:
    z = mp.mpf(str(z))
    phi = mp.exp(-z*z/2) / mp.sqrt(2*mp.pi)
    cdf = mp.erfc(-z/mp.sqrt(2))/2
    log_h.append({'z': float(z), 'log_h': float(mp.log(phi+z*cdf))})
(DEST/'log-h.json').write_text(json.dumps({'precision_digits': 100, 'values': log_h}, indent=2)+'\n')

xs = [[mp.mpf('0.1'), mp.mpf('0.2')], [mp.mpf('0.4'), mp.mpf('0.9')], [mp.mpf('0.8'), mp.mpf('0.5')]]
ys = mp.matrix(['0.3', '-0.7', '0.4'])
ls = [mp.mpf('0.5'), mp.mpf('0.8')]
queries = [[mp.mpf('0.2'), mp.mpf('0.6')], [mp.mpf('0.5'), mp.mpf('0.5')], xs[0]]
models = []
for name in ['rbf', 'matern52']:
    def kernel(a, b):
        r2 = sum(((a[i]-b[i])/ls[i])**2 for i in range(2))
        if name == 'rbf':
            return mp.exp(-r2/2)
        r = mp.sqrt(5*r2)
        return (1+r+5*r2/3)*mp.exp(-r)
    matrix = mp.matrix([[kernel(a,b)+(mp.mpf('2e-10') if i==j else 0)
                         for j,b in enumerate(xs)] for i,a in enumerate(xs)])
    alpha = mp.lu_solve(matrix,ys)
    prior_mu = mp.sqrt(2)+mp.log(2)/2
    prior = sum(mp.log(l)+(mp.log(l)-prior_mu)**2/6 for l in ls)
    nlp = ((ys.T*alpha)[0]+mp.log(mp.det(matrix))+3*mp.log(2*mp.pi))/2+prior
    predictions = []
    for q in queries:
        k = mp.matrix([kernel(x,q) for x in xs])
        predictions.append({'x':list(map(float,q)), 'mean':float((k.T*alpha)[0]),
                            'variance':float(1-(k.T*mp.lu_solve(matrix,k))[0])})
    models.append({'kernel':name,'x':[[float(v) for v in x] for x in xs], 'y':list(map(float,ys)),
                   'lengthscales':list(map(float,ls)), 'negative_log_posterior':float(nlp),'predictions':predictions})
(DEST/'gp.json').write_text(json.dumps({'precision_digits':100,'models':models},indent=2)+'\n')
print('Wrote', DEST)
