SETUP = '''import csv, os, math, time, random
import numpy as np
os.makedirs('data', exist_ok=True); os.makedirs('figures', exist_ok=True)
def save_csv(name, rows):
    with open(os.path.join('data', name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import platform, sklearn; print('Python', platform.python_version(), '| NumPy', np.__version__, '| scikit-learn', sklearn.__version__)'''

D = {}
D['t1_demo'] = '''from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)          # public UCI Wisconsin breast-cancer data (bundled with scikit-learn)
print('records', len(y), '| class counts (0 = malignant, 1 = benign):', np.bincount(y))
rows = []
for noise in (0.0, 0.10):
    yn = y.copy()
    if noise:
        rng = np.random.RandomState(0); flip = rng.rand(len(y)) < noise; yn[flip] = 1 - yn[flip]
    train_acc = faulty_protocol(X, yn, TreeClassifier('entropy'))
    cv_mean, cv_sd = cross_val_accuracy(X, yn, lambda: TreeClassifier('entropy'), k=5, seed=0)
    tr, te = stratified_split(yn, [0.8, 0.2], 42)
    held = accuracy(yn[te], TreeClassifier('entropy').fit(X[tr], yn[tr]).predict(X[te]))
    rows.append(dict(label_noise=noise, train_on_train=train_acc, cv5=cv_mean, cv_sd=cv_sd, held_out=held, optimism_vs_cv=train_acc - cv_mean))
    print(f"label noise {noise:.0%}: faulty protocol accuracy {train_acc:.3f} | 5-fold CV {cv_mean:.3f} (sd {cv_sd:.3f}) | "
          f"held-out {held:.3f} | optimism {100 * (train_acc - cv_mean):.1f} points")
save_csv('task1_optimism.csv', rows)
r = repaired_protocol(X, y, positive=0)
print('Repaired protocol - CV accuracy by max_depth:', {d: round(v[0], 4) for d, v in r['cv'].items()})
print(f"chosen depth {r['best_depth']} | test accuracy {r['test_acc']:.3f} | precision {r['precision']:.3f} | recall {r['recall']:.3f} | "
      f"F1 {r['f1']:.3f} (positive = malignant) | confusion {r['cm']}")
print(f"majority-class baseline: accuracy {r['baseline_acc']:.3f}, F1 for malignant {r['baseline_f1']:.3f}")'''

D['t2_desk'] = '''log = []
tree = id3(DESK, ATTRS, 'Result', log=log)
for depth, n, H, scores, best in log:
    print('  ' * depth + f"node with {n} examples, H = {H}: gains " + ', '.join(f"{a} {g}" for a, g, _ in scores) + f"  -> split on {best}")
print('tree:', tree)
print('training accuracy on the 8 rows:', sum(predict_id3(tree, r) == r['Result'] for r in DESK), '/ 8')'''

D['t2_impl'] = '''from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
tr, va, te = stratified_split(y, [0.6, 0.2, 0.2], 0)
rows = []
for crit in ('entropy', 'gini'):
    t0 = time.perf_counter(); mine = TreeClassifier(crit).fit(X[tr], y[tr]); t_mine = time.perf_counter() - t0
    ref = DecisionTreeClassifier(criterion=crit, random_state=0).fit(X[tr], y[tr])
    pm, pr = mine.predict(X[te]), ref.predict(X[te])
    for name, p, depth, leaves in (('from scratch', pm, mine.depth(), mine.leaves()), ('scikit-learn', pr, ref.get_depth(), ref.get_n_leaves())):
        P, R, F = prf(list(y[te]), list(p), 0)
        rows.append(dict(criterion=crit, model=name, accuracy=accuracy(y[te], p), precision=P, recall=R, f1=F, depth=depth, leaves=leaves))
        print(f"{crit:7} {name:12} acc {accuracy(y[te], p):.3f} prec {P:.3f} rec {R:.3f} F1 {F:.3f} (malignant) depth {depth} leaves {leaves} "
              f"confusion {confusion(list(y[te]), list(p), [0, 1])}")
    print(f"        same prediction on {np.mean(pm == pr):.1%} of the {len(te)} test cases; root split: mine feature {mine.root.feature} "
          f"at {mine.root.threshold:.3f}, sklearn feature {ref.tree_.feature[0]} at {ref.tree_.threshold[0]:.3f}; training time {t_mine:.2f} s")
save_csv('task2_trees.csv', rows)'''

D['t2_prune'] = '''rows = []
for depth in (1, 2, 3, 4, 5, 6, None):
    for leaf in (1, 5):
        m = TreeClassifier('entropy', max_depth=depth, min_samples_leaf=leaf).fit(X[tr], y[tr])
        r = dict(max_depth=depth or 'full', min_leaf=leaf, depth=m.depth(), leaves=m.leaves(),
                 train=accuracy(y[tr], m.predict(X[tr])), val=accuracy(y[va], m.predict(X[va])), test=accuracy(y[te], m.predict(X[te])))
        rows.append(r)
        print(f"max_depth {str(r['max_depth']):4} min_leaf {leaf}: depth {r['depth']} leaves {r['leaves']:2} | train {r['train']:.3f} val {r['val']:.3f} test {r['test']:.3f}")
best = max(rows, key=lambda r: (r['val'], -r['leaves']))
print('chosen on validation:', best)
save_csv('task2_pruning.csv', rows)'''

D['t2_challenge'] = '''changed = [dict(r) for r in DESK]; changed[6]['Result'] = 'Pass'           # row 7: Low, Missed, Good -> Pass
log = []; t2 = id3(changed, ATTRS, 'Result', log=log)
print('row 7 changed to Pass -> root gains:', log[0][3], '-> root', log[0][4]); print('new tree:', t2)
with_id = [dict(r, ID=f"S{r['No']}") for r in DESK]
for crit, mb in (('gain', 1), ('ratio', 1), ('gain', 2)):
    log = []; t3 = id3(with_id, ['ID'] + ATTRS, 'Result', log=log, criterion=crit, min_branch=mb)
    print(f"ID attribute added, criterion {crit}, min branch size {mb}: scores (attr, gain, score) {log[0][3]} -> root {log[0][4]}")
from sklearn.datasets import load_breast_cancer
Xb, yb = load_breast_cancer(return_X_y=True)
trb, teb = stratified_split(yb, [0.8, 0.2], 3)
Xid = np.column_stack([np.arange(len(yb)), Xb])                         # row number as a feature
for leaf in (1, 10):
    m = TreeClassifier('entropy', min_samples_leaf=leaf).fit(Xid[trb], yb[trb])
    print(f"breast-cancer data with a row-ID column, min_samples_leaf {leaf}: root feature {m.root.feature} "
          f"({'ID' if m.root.feature == 0 else 'real feature'}), test accuracy {accuracy(yb[teb], m.predict(Xid[teb])):.3f}")'''

D['t3_trace'] = '''X4 = [1.0, 2.0, 8.0, 9.0]
for rep in (False, True):
    tr = {}; pi, mu, var, h = em(X4, [0.5, 0.5], [2.0, 7.0], [4.0, 4.0], 5, repaired=rep, trace=tr)
    print('REPAIRED' if rep else 'FAULTY', '- first iteration')
    print('   r[i][k] =', np.round(tr['r'], 4).tolist(), '| row sums', np.round(tr['r'].sum(axis=1), 4).tolist())
    print('   Nk =', np.round(tr['Nk'], 4).tolist(), ' mu =', np.round(tr['mu'], 4).tolist(), ' var =', np.round(tr['var'], 4).tolist(), ' pi =', np.round(tr['pi'], 4).tolist())
    print('   log-likelihood per iteration:', [round(x['ll'], 4) for x in h], '| sum of weights:', [round(x['wsum'], 4) for x in h])'''

D['t3_runs'] = '''Xd = make_data()
rows = many_inits(Xd, K=2, T=60, runs=20)
summary = []
for name in ('faulty', 'repaired'):
    fl = [r[name]['final_ll'] for r in rows]
    summary.append(dict(version=name, mean_final_ll=np.mean(fl), runs_with_ll_decrease=sum(r[name]['decreases'] > 0 for r in rows),
                        mean_weight_sum=np.mean([r[name]['wsum'] for r in rows])))
    print(f"{name:8}: mean final log-likelihood {np.mean(fl):12.2f} | runs where LL ever decreased {summary[-1]['runs_with_ll_decrease']:2}/20 | "
          f"mean sum of weights {summary[-1]['mean_weight_sum']:.4g}")
save_csv('task3_em_runs.csv', summary)
from sklearn.mixture import GaussianMixture
g = GaussianMixture(2, random_state=0).fit(Xd.reshape(-1, 1))
rng = np.random.RandomState(1); pi0, mu0, var0 = random_init(Xd, 2, rng)
pi, mu, var, _ = em(Xd, pi0, mu0, var0, 200, True)
o = np.argsort(mu)
print('repaired EM  : pi', np.round(pi[o], 3), 'mu', np.round(mu[o], 3), 'var', np.round(var[o], 3))
og = np.argsort(g.means_.ravel())
print('scikit-learn : pi', np.round(g.weights_[og], 3), 'mu', np.round(g.means_.ravel()[og], 3), 'var', np.round(g.covariances_.ravel()[og], 3))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
pi0, mu0, var0 = np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([4.0, 4.0])
for rep, style in ((False, 'r--'), (True, 'b-')):
    _, _, _, h = em(Xd, pi0, mu0, var0, 40, rep)
    ax[0].plot([x['ll'] for x in h], style, label='repaired' if rep else 'faulty')
    ax[1].plot([x['wsum'] for x in h], style, label='repaired' if rep else 'faulty')
ax[0].set_title('log-likelihood per iteration'); ax[0].set_yscale('symlog'); ax[1].set_title('sum of mixing weights')
for a in ax: a.set_xlabel('iteration'); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig('figures/task3_em.png', dpi=200); plt.show()
print('Regression check - symmetric data X = [-1, -1, 1, 1] (every point has the same total density), one iteration:')
for rep in (False, True):
    pi, mu, var, _ = em([-1.0, -1.0, 1.0, 1.0], [0.5, 0.5], [-0.5, 0.5], [4.0, 4.0], 1, rep)
    print(f"   {'repaired' if rep else 'original'}: mu {np.round(mu, 5)} var {np.round(var, 5)} pi {np.round(pi, 5)}")'''

D['t4_desk'] = '''net = desk_check_network(); x = np.array([[1.0], [0.0]]); y = np.array([[1.0]])
A, cache = net.forward(x)
print('z(1) =', np.round(cache[1][0].ravel(), 4), ' a(1) =', np.round(cache[1][1].ravel(), 4))
print('z(2) =', np.round(cache[2][0].ravel(), 4), ' y_hat = a(2) =', np.round(A.ravel(), 4), ' E = 0.5(y - y_hat)^2 =', round(net.loss_value(A, y), 4))
gW, gb, deltas = net.backward(cache, y)
print('delta(2) =', np.round(deltas[1].ravel(), 5), ' delta(1) =', np.round(deltas[0].ravel(), 5))
print('dE/dW(2) =', np.round(gW[1], 5).tolist(), ' dE/dW(1) =', np.round(gW[0], 5).tolist())
net.step(gW, gb, 0.5)
print('updated W(1) =', np.round(net.W[0], 5).tolist(), ' W(2) =', np.round(net.W[1], 5).tolist(),
      ' b(1) =', np.round(net.b[0].ravel(), 5).tolist(), ' b(2) =', np.round(net.b[1].ravel(), 5).tolist())'''

D['t4_grad'] = '''rng = np.random.RandomState(0)
for sizes, h, o, l in (([3, 4, 2], 'sigmoid', 'sigmoid', 'mse'), ([4, 5, 3], 'relu', 'softmax', 'ce'), ([4, 6, 6, 3], 'sigmoid', 'softmax', 'ce')):
    n = MLP(sizes, h, o, l, seed=1); Xg = rng.randn(sizes[0], 5); Yg = np.eye(sizes[-1])[:, rng.randint(0, sizes[-1], 5)]
    print(f"gradient check {sizes} {h}/{o}/{l}: max relative error {gradient_check(n, Xg, Yg):.2e}")
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); yx = np.array([0, 1, 1, 0])
w, b, p = perceptron(Xx, yx, epochs=100)
print('single-layer perceptron on XOR (100 epochs): predictions', p.tolist(), '-> accuracy', (p == yx).mean())
n = MLP([2, 4, 1], 'sigmoid', 'sigmoid', 'mse', seed=3)
h = train(n, Xx.T.astype(float), yx[None, :].astype(float), 2.0, 5000, batch=4)
print('2-4-1 network on XOR: predictions', n.predict(Xx.T.astype(float)).tolist(), '| outputs', np.round(n.forward(Xx.T.astype(float))[0].ravel(), 3).tolist(),
      '| final loss', round(h['train_loss'][-1], 5))'''

D['t4_grid'] = '''from sklearn.datasets import load_digits
Xd, yd = load_digits(return_X_y=True); Xd = Xd / 16.0          # public 8x8 digits (UCI), bundled with scikit-learn
tr, va, te = stratified_split(yd, [0.6, 0.2, 0.2], 0)
Y = np.eye(10)[yd].T
rows, curves = [], {}
for act in ('sigmoid', 'relu'):
    for hidden in (2, 8, 32, 64):
        for lr in (0.001, 0.01, 0.1, 1.0):
            net = MLP([64, hidden, 10], act, 'softmax', 'ce', seed=0)
            h = train(net, Xd[tr].T, Y[:, tr], lr, 60, batch=32, Xv=Xd[va].T, Yv=Y[:, va])
            p = net.predict(Xd[va].T)
            r = dict(activation=act, hidden=hidden, lr=lr, final_train_loss=h['train_loss'][-1], val_acc=accuracy(list(yd[va]), list(p)),
                     val_macro_f1=macro_f1(list(yd[va]), list(p)))
            rows.append(r); curves[(act, hidden, lr)] = h
            print(f"{act:7} hidden {hidden:2} lr {lr:5}: train loss {r['final_train_loss']:.3f} | val acc {r['val_acc']:.3f} | val macro-F1 {r['val_macro_f1']:.3f}")
save_csv('task4_grid.csv', rows)
best = max(rows, key=lambda r: r['val_macro_f1'])
net = MLP([64, best['hidden'], 10], best['activation'], 'softmax', 'ce', seed=0)
train(net, Xd[tr].T, Y[:, tr], best['lr'], 60, batch=32)
p = net.predict(Xd[te].T)
print('best on validation:', best['activation'], best['hidden'], best['lr'], f"-> TEST accuracy {accuracy(list(yd[te]), list(p)):.3f}, macro-F1 {macro_f1(list(yd[te]), list(p)):.3f}")
print('test confusion matrix (rows = true digit 0-9):'); print(np.array(confusion(list(yd[te]), list(p), list(range(10)))))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for i, act in enumerate(('sigmoid', 'relu')):
    for lr in (0.001, 0.01, 0.1, 1.0):
        ax[i].plot(curves[(act, 32, lr)]['val_loss'], label=f'lr {lr}')
    ax[i].set_title(f'{act}, 32 hidden units: validation loss'); ax[i].set_xlabel('epoch'); ax[i].set_yscale('log'); ax[i].legend(fontsize=8); ax[i].grid(alpha=.3)
plt.tight_layout(); plt.savefig('figures/task4_curves.png', dpi=200); plt.show()'''

D['t4_fail'] = '''print('Failure 1 - learning rate far too large: ReLU, 64 hidden, learning rate 5.0')
net = MLP([64, 64, 10], 'relu', 'softmax', 'ce', seed=0)
h = train(net, Xd[tr].T, Y[:, tr], 5.0, 10, batch=32, record_grads=True)
for ep, (l, g) in enumerate(zip(h['train_loss'], h['grad_norms'])):
    print(f"   epoch {ep + 1}: train loss {l:10.4f} | gradient norm per layer (W1, W2) {np.round(g, 3).tolist()}")
print('Failure 2 - vanishing gradient: 6 sigmoid hidden layers of 32, lr 0.1')
deep = MLP([64, 32, 32, 32, 32, 32, 32, 10], 'sigmoid', 'softmax', 'ce', seed=0)
h = train(deep, Xd[tr].T, Y[:, tr], 0.1, 30, batch=32, record_grads=True)
for ep in (0, 9, 29):
    print(f"   epoch {ep + 1}: loss {h['train_loss'][ep]:.3f} | grad norm layer 1..7 {np.round(h['grad_norms'][ep], 5).tolist()}")
print('   val accuracy after 30 epochs:', round(accuracy(list(yd[va]), list(deep.predict(Xd[va].T))), 3))
fix = MLP([64, 32, 32, 32, 32, 32, 32, 10], 'relu', 'softmax', 'ce', seed=0)
h2 = train(fix, Xd[tr].T, Y[:, tr], 0.1, 30, batch=32, record_grads=True)
print('   same depth with ReLU + He initialisation: grad norms epoch 1', np.round(h2['grad_norms'][0], 4).tolist(),
      '| val accuracy', round(accuracy(list(yd[va]), list(fix.predict(Xd[va].T))), 3))'''

D['t5_nb'] = '''toy = toy_zero_count()
for a in (0, 1):
    print(f"toy table, alpha={a}: query (Load=High, Fan=off) ->", {c: dict(zip(('prior', 'P(load|c)', 'P(fan|c)', 'score'), v)) for c, v in toy[a].items()})
print('query (Load=Low, Fan=off) with alpha=0 -> scores', toy['unseen'], '(every class impossible: no decision)')
Xg, yg = substation_data()
print('substation records', len(yg), '| hazardous', int(yg.sum()), '| features', FEATURES)
tr, va, te = stratified_split(yg, [0.6, 0.2, 0.2], 1)
rows = []
for ntrain in (len(tr), 200):
    sub = tr if ntrain == len(tr) else np.random.RandomState(0).choice(tr, ntrain, replace=False)
    for a in (0, 0.1, 1, 10):
        nb = NaiveBayes(a, bins=20).fit(Xg[sub], yg[sub])
        prob = nb.predict_proba(Xg[te]); pred = nb.predict(Xg[te])
        zero = float(np.isinf(nb.joint_log(Xg[te])).any(axis=1).mean())
        P, R, F = prf(list(yg[te]), list(pred), 1)
        rows.append(dict(train_size=len(sub), alpha=a, accuracy=accuracy(list(yg[te]), list(pred)), precision=P, recall=R, f1=F,
                         log_loss=log_loss(yg[te], prob[:, 1]), test_rows_with_zero=zero))
        print(f"train {len(sub):4} alpha {a:4}: acc {rows[-1]['accuracy']:.3f} prec {P:.3f} rec {R:.3f} F1 {F:.3f} | log-loss {rows[-1]['log_loss']:.4f} | "
              f"test rows hitting a zero probability {zero:.1%}")
save_csv('task5_naive_bayes.csv', rows)'''

D['t5_svm'] = '''from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
trv = np.concatenate([tr, va])
rows = []
for kernel, grid in (('linear', {'svc__C': [0.01, 0.1, 1, 10, 100]}), ('rbf', {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': [0.01, 0.1, 1, 10]})):
    t0 = time.perf_counter()
    g = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel=kernel)), grid, cv=5).fit(Xg[trv], yg[trv])
    secs = time.perf_counter() - t0
    pred = g.predict(Xg[te]); P, R, F = prf(list(yg[te]), list(pred), 1)
    rows.append(dict(model=f'SVM {kernel}', params=str(g.best_params_), support_vectors=int(g.best_estimator_[-1].n_support_.sum()),
                     accuracy=accuracy(list(yg[te]), list(pred)), f1=F, search_seconds=secs))
nb = NaiveBayes(1.0, bins=20).fit(Xg[trv], yg[trv]); pred = nb.predict(Xg[te])
rows.append(dict(model='Naive Bayes (alpha 1)', params='20 bins', support_vectors='-', accuracy=accuracy(list(yg[te]), list(pred)), f1=prf(list(yg[te]), list(pred), 1)[2], search_seconds='-'))
depths = {d: accuracy(list(yg[va]), list(TreeClassifier('entropy', max_depth=d).fit(Xg[tr], yg[tr]).predict(Xg[va]))) for d in (2, 3, 4, 5, 6, 8, None)}
bd = max(depths, key=lambda d: (depths[d], -(d or 99)))
tree = TreeClassifier('entropy', max_depth=bd).fit(Xg[trv], yg[trv]); pred = tree.predict(Xg[te])
rows.append(dict(model=f'Task 2 tree (depth {bd})', params='depth chosen on validation', support_vectors='-', accuracy=accuracy(list(yg[te]), list(pred)),
                 f1=prf(list(yg[te]), list(pred), 1)[2], search_seconds='-'))
for r in rows:
    print(f"{r['model']:24} {r['params']:36} SVs {str(r['support_vectors']):4} | test acc {r['accuracy']:.3f} | F1 {r['f1']:.3f} | search time {r['search_seconds'] if isinstance(r['search_seconds'], str) else round(r['search_seconds'], 2)} s")
save_csv('task5_models.csv', rows)
import matplotlib.pyplot as plt
cols = [0, 3]                                          # load_pct, voltage_pu
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
xx, yy = np.meshgrid(np.linspace(40, 140, 300), np.linspace(0.8, 1.2, 300))
for a, kernel in zip(ax, ('linear', 'rbf')):
    m = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=10, gamma='scale')).fit(Xg[trv][:, cols], yg[trv])
    zz = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    a.contourf(xx, yy, zz, alpha=0.25, cmap='coolwarm')
    a.scatter(Xg[te][:, 0], Xg[te][:, 3], c=yg[te], cmap='coolwarm', s=8, edgecolor='k', linewidth=0.2)
    acc = accuracy(list(yg[te]), list(m.predict(Xg[te][:, cols])))
    a.set_title(f'{kernel} SVM on (load, voltage): test acc {acc:.3f}, {m[-1].n_support_.sum()} SVs'); a.set_xlabel('load %'); a.set_ylabel('voltage (p.u.)')
plt.tight_layout(); plt.savefig('figures/task5_boundaries.png', dpi=200); plt.show()'''

D['t5_ebl'] = '''for name, head, body in THEORY:
    print(f"{name}: {show_t(head)} <= {' AND '.join(show_t(b) for b in body)}")
st = Stats(); s, tree = next(prove(('Hazard', 'S17'), TRAIN_EXAMPLE, THEORY, st))
print('Explanation of the training case S17:'); print('\\n'.join('   ' + l for l in show_tree(tree)))
learned = generalise('Hazard', tree, THEORY)
print('LEARNED RULE:', show_t(learned[1]), '<=', ' AND '.join(show_t(b) for b in learned[2]))
rows = evaluate_cases(learned)
for r in rows:
    print(f"{r['case']:3}: domain theory {str(r['theory']):5} ({r['theory_goals']:2} goals, {r['theory_unify']:3} unifications) | "
          f"learned rule {str(r['learned']):5} ({r['learned_goals']} goals, {r['learned_unify']} unifications)")
save_csv('task5_ebl.csv', rows)
fired = [r for r in rows if r['learned']]
print(f"hazard by theory {sum(r['theory'] for r in rows)}/20 | learned rule fires {len(fired)}/20 | learned but not theory {sum(r['learned'] and not r['theory'] for r in rows)}")
print(f"on the cases where the rule fires: {np.mean([r['learned_unify'] for r in fired]):.1f} vs {np.mean([r['theory_unify'] for r in fired]):.1f} unifications; "
      f"over all 20 cases: {np.mean([r['learned_unify'] for r in rows]):.1f} vs {np.mean([r['theory_unify'] for r in rows]):.1f}")'''

D['t6_desk'] = '''log, Q = toy_desk_check()
print('alpha = 0.5, gamma = 0.9; Q starts at 0')
for i, (s, a, r, s2, old, best, new) in enumerate(log, 1):
    print(f"update {i}: {s:4} --{a:3}--> reward {r:4} next {s2:4} | Q old {old:6.3f} + 0.5*({r} + 0.9*{best:.3f} - {old:.3f}) = {new:7.3f}")
print('final Q table:', {k: round(v, 3) for k, v in Q.items()})'''

D['t6_train'] = '''F = ('until', 'part', 'T')
base = evaluate(thermostat, days=30, seed=500)
rows, curves = [], []
for seed in range(20):
    Q, curve = q_learning(seed, 15000, features=F)
    r = evaluate(greedy_policy(Q, F), days=30, seed=500)
    r.update(seed=seed, states=len(Q), episodes_to_converge=episodes_to_converge(curve))
    rows.append(r); curves.append(curve)
save_csv('task6_seeds.csv', rows)
def ms(k): v = [r[k] for r in rows]; return f"{np.mean(v):8.2f} +/- {np.std(v):5.2f}"
print('per test day (30 unseen days)   Q-learning (20 seeds)        fixed-rule thermostat')
for k, label in (('reward', 'total reward'), ('cost', 'energy cost (Rs)'), ('kwh', 'energy (kWh)'), ('violations', 'comfort violations (slots)'),
                 ('viol_deg', 'violation degree-slots'), ('arrival_viol', 'violations at arrival')):
    print(f"{label:28} {ms(k)}        {base[k]:8.2f}")
print('episodes to convergence:', ms('episodes_to_converge'), '| states visited:', ms('states'))
_, trace_q = evaluate(greedy_policy(*[q_learning(0, 15000, features=F)[0]], F), days=1, seed=777, log_day=0)
print('one test day, Q-learning policy (hour, T, occupied, price, action) from 08:00 to 10:00:', [t for t in trace_q if 8.0 <= t[0] < 10.0])
import matplotlib.pyplot as plt
C = np.array(curves); sm = np.array([np.convolve(c, np.ones(200) / 200, mode='valid') for c in C])
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.plot(sm.mean(axis=0), label='Q-learning, mean of 20 seeds'); ax.fill_between(range(sm.shape[1]), sm.min(axis=0), sm.max(axis=0), alpha=0.2)
ax.axhline(base['reward'], color='red', ls='--', label='fixed-rule thermostat')
ax.set_xlabel('episode (training day)'); ax.set_ylabel('reward per day (200-episode mean)'); ax.legend(); ax.grid(alpha=.3); ax.set_ylim(-700, -150)
plt.tight_layout(); plt.savefig('figures/task6_learning.png', dpi=200); plt.show()'''

D['t6_attack'] = '''Q, _ = q_learning(0, 15000, features=F)
pol = greedy_policy(Q, F)
print('ATTACK 1 - unseen price pattern: Rs 25/kWh from 08:00 to 09:00 (training never saw it)')
for name, p in (('Q-learning', pol), ('thermostat', thermostat)):
    n, s = evaluate(p, days=30, seed=500), evaluate(p, days=30, seed=500, price_pattern='spike')
    print(f"   {name:10}: cost normal {n['cost']:7.2f} -> spike {s['cost']:7.2f} | violations {n['violations']:.2f} -> {s['violations']:.2f}")
print('ATTACK 2 - remove the timetable feature (state = day part, temperature, occupied) - can the agent still pre-cool?')
F2 = ('part', 'T', 'occupied')
res = []
for seed in range(5):
    Q2, _ = q_learning(seed, 15000, features=F2); res.append(evaluate(greedy_policy(Q2, F2), days=30, seed=500))
res1 = [evaluate(greedy_policy(q_learning(seed, 15000, features=F)[0], F), days=30, seed=500) for seed in range(5)]
for label, R in (('with timetable feature', res1), ('without timetable feature', res)):
    print(f"   {label:26}: cost {np.mean([r['cost'] for r in R]):7.2f} | violations {np.mean([r['violations'] for r in R]):5.2f} | "
          f"at arrival {np.mean([r['arrival_viol'] for r in R]):.2f} | reward {np.mean([r['reward'] for r in R]):8.2f}")
Q2, _ = q_learning(0, 15000, features=F2)
_, tr2 = evaluate(greedy_policy(Q2, F2), days=1, seed=777, log_day=0)
print('   no-timetable agent, 08:00-09:45:', [t for t in tr2 if 8.0 <= t[0] < 9.75])'''

ORDER = [
    ('Task 1 and 2 - protocol and decision trees (from scratch)', 'u4_tree.py', []),
    ('Task 1 - faulty versus repaired protocol', 'u4_protocol.py', ['t1_demo']),
    ('Task 2 - desk-check, implementation, pruning, challenge', None, ['t2_desk', 't2_impl', 't2_prune', 't2_challenge']),
    ('Task 3 - EM forensics', 'u4_em.py', ['t3_trace', 't3_runs']),
    ('Task 4 - back-propagation', 'u4_nn.py', ['t4_desk', 't4_grad', 't4_grid', 't4_fail']),
    ('Task 5.1 / 5.2 - Naive Bayes and kernel machines', 'u4_grid.py', ['t5_nb', 't5_svm']),
    ('Task 5.3 - explanation-based learning', 'u4_fol.py', []),
    ('Task 5.3 - EBL', 'u4_ebl.py', ['t5_ebl']),
    ('Task 6 - reinforcement-learning controller', 'u4_rl.py', ['t6_desk', 't6_train', 't6_attack']),
]
