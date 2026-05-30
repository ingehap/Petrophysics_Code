"""
Article 2: Multiple Well-Log Depth Matching Using Deep Q-Learning
Bittar, Wang, Wu, Chen (2021)
DOI: 10.30632/PJV62N4-2021a1

Frames well-log depth matching as a Markov Decision Process solved with a
(Deep) Q-Network: an agent slides a reference-log window left/right over a
target log by choosing discrete shift actions, and triggers a "stop" action
at the match point.  The reward drives the agent toward the premarked match.

Implements:

  - Bellman / Q-learning update
        Q(s,a) <- Q(s,a) + alpha*(r + gamma*max_a' Q(s',a') - Q(s,a))
  - epsilon-greedy action selection
  - the depth-matching MDP (state = agent position; actions = shifts + stop)
  - tabular training + greedy evaluation

Note: the paper's display equations were image-rendered and not in the text,
and it uses a deep CNN Q-network (Rainbow DQN); here the SAME MDP and Bellman
update are solved with a compact tabular Q-learner so the module runs with
numpy alone.  Hyperparameter defaults follow the paper (gamma=0.99).
"""

import numpy as np

# Discrete shift actions in samples; 0 is the "stop / match found" action
ACTIONS = [-10, -5, -1, 0, +1, +5, +10]
STOP = ACTIONS.index(0)


# ---------------------------------------------- Bellman update ----------

def bellman_update(Q, s, a, r, s2, alpha=0.1, gamma=0.99):
    """Tabular Q-learning update toward r + gamma*max_a' Q(s2,a')."""
    target = r + gamma * np.max(Q[s2])
    Q[s, a] += alpha * (target - Q[s, a])
    return Q[s, a]


# ---------------------------------------------- policy ------------------

def epsilon_greedy(Q, s, epsilon, rng):
    """Pick a random action with prob. epsilon, else argmax_a Q(s,a)."""
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTIONS)))
    return int(np.argmax(Q[s]))


# ---------------------------------------------- reward / step -----------

def step(pos, action_idx, n, match, tol):
    """Apply a shift action; return (new_pos, reward, done)."""
    a = ACTIONS[action_idx]
    if a == 0:                                   # stop action
        done = True
        reward = 10.0 if abs(pos - match) <= tol else -10.0
        return pos, reward, done
    new_pos = int(np.clip(pos + a, 0, n))
    # progress reward: positive when the move reduces distance to the match
    reward = float(abs(pos - match) - abs(new_pos - match)) - 0.1
    return new_pos, reward, False


def train(n, match, tol=2, episodes=6000, alpha=0.2, gamma=0.99,
          max_steps=80, seed=0):
    """Tabular Q-learning on the depth-matching MDP.  Returns the Q-table."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((n + 1, len(ACTIONS)))
    eps = 1.0
    for _ in range(episodes):
        pos = int(rng.integers(0, n + 1))
        for _ in range(max_steps):
            a = epsilon_greedy(Q, pos, eps, rng)
            new_pos, r, done = step(pos, a, n, match, tol)
            s2 = new_pos
            bellman_update(Q, pos, a, r, s2, alpha, gamma)
            pos = new_pos
            if done:
                break
        eps = max(0.05, eps - 1.0 / episodes)
    return Q


def greedy_match(Q, start, n, match, max_steps=80):
    """Roll out the greedy policy; return (final_pos, stopped)."""
    pos = start
    for _ in range(max_steps):
        a = int(np.argmax(Q[pos]))
        if ACTIONS[a] == 0:
            return pos, True
        pos = int(np.clip(pos + ACTIONS[a], 0, n))
    return pos, False


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Well-Log Depth Matching via Deep Q-Learning")
    print("=" * 60)

    n, match, tol = 120, 73, 2
    Q = train(n, match, tol=tol, episodes=6000, seed=0)

    # Greedy rollouts from several starts should reach and stop near the match
    rng = np.random.default_rng(1)
    starts = rng.integers(0, n + 1, size=10)
    finals = []
    n_stop = 0
    for s0 in starts:
        pos, stopped = greedy_match(Q, int(s0), n, match)
        finals.append(abs(pos - match))
        n_stop += int(stopped)
    max_err = max(finals)
    print(f"  match point            = {match}")
    print(f"  max |pos-match| (10 starts) = {max_err}")
    print(f"  stop action triggered  = {n_stop}/10")
    assert max_err <= tol + 1, "greedy policy should reach the match point"
    assert n_stop >= 9, "agent should trigger the stop action"

    # Bellman update moves Q toward the target
    Qt = np.zeros((3, len(ACTIONS)))
    before = Qt[0, 1]
    bellman_update(Qt, 0, 1, r=5.0, s2=1, alpha=0.5, gamma=0.99)
    assert Qt[0, 1] > before
    print("  PASS")
    return {"max_err": int(max_err), "n_stop": n_stop}


if __name__ == "__main__":
    test_all()
