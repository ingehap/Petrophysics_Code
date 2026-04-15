"""
Article 3: Use of Symbolic Regression for Developing Petrophysical
Interpretation Models
Chen, Shao, Sheng, and Kwak (2023)
DOI: 10.30632/PJV64N2-2023a3

Implements:
  - Pearson and Spearman correlation heatmaps for feature selection
  - A simple genetic-programming symbolic regression engine
  - Ensemble of SR equations
  - Model-discrimination scoring (mathematical complexity vs. R^2)
"""

import numpy as np
import operator
import random
from copy import deepcopy
from scipy.stats import pearsonr, spearmanr


# ----------------------------------------------- correlation heatmaps ---

def pearson_heatmap(X, names=None):
    """Pearson correlation matrix."""
    n = X.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = pearsonr(X[:, i], X[:, j])[0]
    return M


def spearman_heatmap(X, names=None):
    n = X.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = spearmanr(X[:, i], X[:, j])[0]
    return M


def select_top_features(X, y, k=3, method="pearson"):
    """Return indices of top-k features most correlated with y."""
    n = X.shape[1]
    fn = pearsonr if method == "pearson" else spearmanr
    scores = np.array([abs(fn(X[:, i], y)[0]) for i in range(n)])
    return np.argsort(scores)[-k:][::-1], scores


# -------------------------------------------- symbolic regression ---

def safe_div(a, b):
    return np.divide(a, b, out=np.ones_like(a, dtype=float), where=np.abs(b) > 1e-9)

def safe_log(a):
    return np.log(np.abs(a) + 1e-9)

def safe_sqrt(a):
    return np.sqrt(np.abs(a))

OPS = {
    "+": (2, np.add),
    "-": (2, np.subtract),
    "*": (2, np.multiply),
    "/": (2, safe_div),
    "log": (1, safe_log),
    "sqrt": (1, safe_sqrt),
    "sq": (1, np.square),
}


class Node:
    def __init__(self, kind, value=None, children=None):
        # kind in {"op", "var", "const"}
        self.kind = kind
        self.value = value
        self.children = children or []

    def __str__(self):
        if self.kind == "var":
            return f"x{self.value}"
        if self.kind == "const":
            return f"{self.value:.3g}"
        # op
        if len(self.children) == 1:
            return f"{self.value}({self.children[0]})"
        return f"({self.children[0]} {self.value} {self.children[1]})"

    def evaluate(self, X):
        if self.kind == "var":
            return X[:, self.value]
        if self.kind == "const":
            return np.full(X.shape[0], float(self.value))
        arity, fn = OPS[self.value]
        if arity == 1:
            return fn(self.children[0].evaluate(X))
        return fn(self.children[0].evaluate(X), self.children[1].evaluate(X))

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def all_nodes(self):
        out = [self]
        for c in self.children:
            out.extend(c.all_nodes())
        return out


def random_tree(n_vars, depth, rng, max_depth=4):
    if depth >= max_depth or (depth > 0 and rng.random() < 0.3):
        if rng.random() < 0.5:
            return Node("var", rng.integers(0, n_vars))
        return Node("const", rng.uniform(-2.0, 2.0))
    op = rng.choice(list(OPS.keys()))
    arity = OPS[op][0]
    children = [random_tree(n_vars, depth + 1, rng, max_depth) for _ in range(arity)]
    return Node("op", op, children)


def crossover(a, b, rng):
    a, b = deepcopy(a), deepcopy(b)
    nodes_a = a.all_nodes()
    nodes_b = b.all_nodes()
    na = nodes_a[rng.integers(0, len(nodes_a))]
    nb = nodes_b[rng.integers(0, len(nodes_b))]
    # swap by mutation: rebuild node a's slot to nb
    # quick swap via attribute copy
    na.kind, nb.kind = nb.kind, na.kind
    na.value, nb.value = nb.value, na.value
    na.children, nb.children = nb.children, na.children
    return a


def mutate(tree, n_vars, rng, max_depth=4):
    t = deepcopy(tree)
    nodes = t.all_nodes()
    target = nodes[rng.integers(0, len(nodes))]
    new = random_tree(n_vars, 0, rng, max_depth=max_depth)
    target.kind = new.kind
    target.value = new.value
    target.children = new.children
    return t


def fitness(tree, X, y, complexity_penalty=0.005):
    try:
        pred = tree.evaluate(X)
    except Exception:
        return 1e10
    if not np.all(np.isfinite(pred)):
        return 1e10
    mse = np.mean((pred - y) ** 2)
    return mse + complexity_penalty * tree.size()


def symbolic_regression(X, y, pop_size=80, n_gen=40, seed=0, max_depth=4):
    """A minimalist GP-based symbolic regression."""
    rng = np.random.default_rng(seed)
    n_vars = X.shape[1]
    population = [random_tree(n_vars, 0, rng, max_depth) for _ in range(pop_size)]
    best = None
    best_fit = np.inf
    for gen in range(n_gen):
        fits = [fitness(t, X, y) for t in population]
        order = np.argsort(fits)
        if fits[order[0]] < best_fit:
            best_fit = fits[order[0]]
            best = deepcopy(population[order[0]])
        # tournament selection
        new_pop = [deepcopy(population[order[i]]) for i in range(5)]  # elitism
        while len(new_pop) < pop_size:
            i, j = rng.integers(0, pop_size, 2)
            parent1 = population[i if fits[i] < fits[j] else j]
            i, j = rng.integers(0, pop_size, 2)
            parent2 = population[i if fits[i] < fits[j] else j]
            child = crossover(parent1, parent2, rng)
            if rng.random() < 0.3:
                child = mutate(child, n_vars, rng, max_depth)
            new_pop.append(child)
        population = new_pop
    return best, best_fit


def ensemble_predict(trees, X):
    return np.mean([t.evaluate(X) for t in trees], axis=0)


# ------------------------------------------------------------- testing ---

def synthetic_data(seed=0):
    """
    Synthetic petrophysical data resembling Archie F = phi^-m, plus
    extra logs as decoys.
    """
    rng = np.random.default_rng(seed)
    n = 200
    phi = rng.uniform(0.05, 0.30, n)
    vp = rng.uniform(2500, 5500, n)
    t2lm = rng.uniform(10, 200, n)
    decoy = rng.normal(0, 1, n)
    F = phi ** -2.0 * np.exp(rng.normal(0, 0.05, n))
    X = np.column_stack([phi, vp, t2lm, decoy])
    return X, F, ["phi", "vp", "t2lm", "decoy"]


def test_all():
    print("=" * 60)
    print("Article 3: Symbolic Regression for Petrophysical Models")
    print("=" * 60)
    X, y, names = synthetic_data()

    # heatmaps
    Pmat = pearson_heatmap(X)
    Smat = spearman_heatmap(X)
    print(f"  Pearson [phi,F]: {pearsonr(X[:, 0], y)[0]:.3f}  "
          f"Spearman: {spearmanr(X[:, 0], y)[0]:.3f}")

    # feature selection
    top, scores = select_top_features(X, y, k=2, method="spearman")
    print(f"  Top features (spearman): {[names[i] for i in top]}")

    # SR — log-target makes the search easier
    y_log = np.log(y)
    tree, fit = symbolic_regression(X, y_log, pop_size=60, n_gen=30, seed=1)
    pred = tree.evaluate(X)
    r2 = 1 - np.sum((y_log - pred) ** 2) / np.sum((y_log - y_log.mean()) ** 2)
    print(f"  Best equation: {tree}")
    print(f"  Equation size: {tree.size()}    R^2 (log F) = {r2:.3f}")

    # ensemble of 3 different seeds
    trees = [symbolic_regression(X, y_log, pop_size=40, n_gen=20, seed=s)[0]
             for s in range(3)]
    pred_ens = ensemble_predict(trees, X)
    r2_ens = 1 - np.sum((y_log - pred_ens) ** 2) / np.sum((y_log - y_log.mean()) ** 2)
    print(f"  Ensemble R^2:  {r2_ens:.3f}")

    assert r2 > 0.3 or r2_ens > 0.3, "SR failed to learn anything"
    print("  PASS")
    return {"r2_single": r2, "r2_ens": r2_ens}


if __name__ == "__main__":
    test_all()
