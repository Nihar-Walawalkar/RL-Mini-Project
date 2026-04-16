import numpy as np
from snake_env import SnakeEnv
from agents import QLearningAgent, SARSAAgent, DoubleQLearningAgent

def train_agent(agent, episodes=1000, grid_size=10, callback=None):
    env = SnakeEnv(grid_size)
    rewards_per_episode = []
    scores_per_episode  = []
    steps_per_episode   = []
    epsilons            = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        scores_per_episode.append(env.score)
        steps_per_episode.append(env.steps)
        epsilons.append(agent.epsilon)

        if callback:
            callback(ep + 1, total_reward, env.score)

    return {
        "rewards": rewards_per_episode,
        "scores":  scores_per_episode,
        "steps":   steps_per_episode,
        "epsilons": epsilons,
        "q_table_size": agent.get_q_table_size(),
        "agent": agent,
    }


def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid').tolist()


def run_episode_visual(agent, grid_size=10):
    """Run one greedy episode and return frames + death reason."""
    env = SnakeEnv(grid_size)
    state = env.reset()
    frames = [env.get_grid().copy()]
    done = False
    old_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy
    death_reason = "timeout"

    while not done:
        head_before = env.snake[0]
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        frames.append(env.get_grid().copy())
        if done:
            head = env.snake[0] if env.snake else head_before
            r, c = env._next_pos(head_before, env.direction)
            if r < 0 or r >= env.grid_size or c < 0 or c >= env.grid_size:
                death_reason = "wall"
            elif (r, c) in list(env.snake)[1:]:
                death_reason = "self"
            else:
                death_reason = "timeout"
        state = next_state
        if len(frames) > 500:
            break

    agent.epsilon = old_eps
    return frames, env.score, death_reason
