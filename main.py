# main.py
"""
Module: main.py
Description: Orchestrates the multi-agent storytelling simulation.
Initializes agents and Director, runs the storytelling loop (with or without training), and evaluates the results.
"""
from director import StoryEnv, DirectorAgent
#from utils import evaluate_story, prompt_human_evaluation, log_episode
from utils import evaluate_story

def main(train_director: bool = False, train_steps: int = 5000, collect_metrics: bool = False):
    """
    Run the collaborative storytelling simulation.
    If train_director is True, it will train the Director agent using PPO for given timesteps.
    Then it generates a story using the (trained or default) director policy.
    """
    # Initialize the storytelling environment and director
    if train_director:
        print("Training Director agent with PPO for", train_steps, "timesteps...")
        env = StoryEnv(training=train_director)
        director = DirectorAgent(env)
        director.train(timesteps=train_steps)
        director.save_policy("director_model")
        return
    else:
        print("Calling Director agent without training for", train_steps, "timesteps...")
        env = StoryEnv(training=False)
        director = DirectorAgent(env)      
        director.load_policy("director_model")
    obs, info = env.reset()
    done = False
    while not done:
        action = director.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    # Retrieve the final story and reward
    story = env.story
    total_reward = info.get('episode_reward', 0.0)
    print("=== Final Story ===")
    print(story)
    print("===================")
    # Evaluate the story with metrics
    metrics = evaluate_story(story)
    print("Story Metrics:", metrics)

    return metrics if collect_metrics else None

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind

    NUM_RUNS = 10
    ppo_metrics = []
    untrained_metrics = []

    print("\n=== Running PPO-trained Director experiments ===")
    main(train_director=True, train_steps=10000)

    for _ in range(NUM_RUNS):
        metrics = main(train_director=False, collect_metrics=True)
        ppo_metrics.append(metrics)
    
    print("\n=== Running Untrained Director experiments ===")
    for _ in range(NUM_RUNS):
        metrics = main(train_director=False, collect_metrics=True)
        untrained_metrics.append(metrics)

    # Extract values
    def extract(metric_list, key):
        return [m[key] for m in metric_list]

    length_ppo = extract(ppo_metrics, "length")
    length_untrained = extract(untrained_metrics, "length")
    char_ppo = extract(ppo_metrics, "num_characters")
    char_untrained = extract(untrained_metrics, "num_characters")
    fantasy_ppo = extract(ppo_metrics, "num_fantasy_elements")
    fantasy_untrained = extract(untrained_metrics, "num_fantasy_elements")

    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind

    # T-tests
    def report_ttest(label, x, y):
        t, p = ttest_ind(x, y)
        print(f"{label}: t={t:.2f}, p={p:.2e}")

    print("\n=== T-Test Results ===")
    report_ttest("Length", length_ppo, length_untrained)
    report_ttest("Characters", char_ppo, char_untrained)
    report_ttest("Fantasy Elements", fantasy_ppo, fantasy_untrained)

    # Plotting
    plt.figure(figsize=(12, 4))
    for i, (label, ppo, untrained) in enumerate([
        ("Total Reward", length_ppo, length_untrained),
        ("Characters", char_ppo, char_untrained),
        ("Fantasy Elements", fantasy_ppo, fantasy_untrained),
    ]):
        plt.subplot(1, 3, i + 1)
        plt.boxplot([ppo, untrained], labels=["PPO", "Untrained"])
    plt.title(label)
    plt.tight_layout()
    plt.show()
