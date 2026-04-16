import io
import pickle
import numpy as np
from collections import defaultdict


class BaseAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, n_actions=3):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions     = n_actions
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_size(self):
        return len(self.Q)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, buf):
        """Serialise Q-table into a binary buffer via pickle."""
        pickle.dump(
            {"Q": dict(self.Q), "n_actions": self.n_actions},
            buf,
        )

    def load(self, buf):
        """Restore Q-table from a binary buffer."""
        data = pickle.load(buf)
        n    = data.get("n_actions", self.n_actions)
        self.Q = defaultdict(
            lambda: np.zeros(n),
            {k: np.array(v) for k, v in data["Q"].items()},
        )


class QLearningAgent(BaseAgent):
    """Off-policy TD control."""
    name  = "Q-Learning"
    color = "#6366f1"

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
        self.decay_epsilon()


class SARSAAgent(BaseAgent):
    """On-policy TD control."""
    name  = "SARSA"
    color = "#10b981"

    def update(self, state, action, reward, next_state, done):
        next_action = self.choose_action(next_state)
        target = reward if done else reward + self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
        self.decay_epsilon()


class DoubleQLearningAgent(BaseAgent):
    """Double Q-Learning — reduces overestimation bias with two Q-tables."""
    name  = "Double Q-Learning"
    color = "#f59e0b"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state] + self.Q2[state]))

    def update(self, state, action, reward, next_state, done):
        if np.random.rand() < 0.5:
            best_next = int(np.argmax(self.Q[next_state]))
            target = reward if done else reward + self.gamma * self.Q2[next_state][best_next]
            self.Q[state][action] += self.alpha * (target - self.Q[state][action])
        else:
            best_next = int(np.argmax(self.Q2[next_state]))
            target = reward if done else reward + self.gamma * self.Q[next_state][best_next]
            self.Q2[state][action] += self.alpha * (target - self.Q2[state][action])
        self.decay_epsilon()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, buf):
        """Serialise both Q-tables into a binary buffer via pickle."""
        pickle.dump(
            {
                "Q":        dict(self.Q),
                "Q2":       dict(self.Q2),
                "n_actions": self.n_actions,
            },
            buf,
        )

    def load(self, buf):
        """Restore both Q-tables from a binary buffer."""
        data = pickle.load(buf)
        n    = data.get("n_actions", self.n_actions)
        self.Q  = defaultdict(
            lambda: np.zeros(n),
            {k: np.array(v) for k, v in data["Q"].items()},
        )
        self.Q2 = defaultdict(
            lambda: np.zeros(n),
            {k: np.array(v) for k, v in data["Q2"].items()},
        )
