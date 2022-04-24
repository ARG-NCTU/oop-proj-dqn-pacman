from deep_Q_network import *
from utils import *

optimization = lambda it, r: it % K_FRAME and r  # or r in (-10, 50, 200)

episodes = 0
learn_counter = 0
last_action = 0
best_score = 0


# one_game = [] # useful to save a video

# Main loop
while True:
    if dmaker.steps_done > MAX_FRAMES:
        break
    episodes += 1

    obs = env.reset()
    lives = 3
    jump_dead_step = False

    # Avoid beginning steps of the game
    for i_step in range(AVOIDED_STEPS - 5):
        obs, reward, done, info = env.step(0)

    observations = init_obs(env)
    obs, reward, done, info = env.step(0)
    state = preprocessing_observation(observations, obs)

    got_reward = False

    # one_game.clear()

    no_move_count = 0
    while True:
        if dmaker.steps_done > MAX_FRAMES:
            break
        # epsilon greedy decision maker
        action = dmaker.select_action(state, policy_DQN, display, learn_counter)
        action_ = action.item()

        obs, reward_, done, info = env.step(action_)
        display.obs = obs
        # one_game.append(obs)
        reward = transform_reward(reward_)

        if info["lives"] < lives:
            lives -= 1
            jump_dead_step = True
            got_reward = False
            reward += REWARDS["lose"]

        if done and lives > 0:
            reward += REWARDS["win"]

        last_action = action_
        got_reward = got_reward or reward != 0
        display.add_reward(reward)
        reward = torch.tensor([reward], device=device)

        next_state = preprocessing_observation(observations, obs)

        if got_reward:
            memory.push(state, action, reward, next_state, done)

        state = next_state
        if optimization(dmaker.steps_done, got_reward):
            learn_counter = optimize_model(
                policy_DQN, target_DQN, memory, optimizer, display, learn_counter, device
            )

        display.show()
        if done:
            display.successes += info["lives"] > 0
            break
        if jump_dead_step:
            for i_dead in range(DEAD_STEPS):
                obs, reward, done, info = env.step(0)
            jump_dead_step = False

    if dmaker.steps_done % TARGET_UPDATE == 0:
        target_DQN.load_state_dict(policy_DQN.state_dict())

    if episodes % SAVE_MODEL == 0:
        torch.save(policy_DQN.state_dict(), PATH_MODELS / f"policy-model-{episodes}.pt")
        torch.save(target_DQN.state_dict(), PATH_MODELS / f"target-model-{episodes}.pt")
        display.save()

    display.new_episode()

torch.save(policy_DQN.state_dict(), PATH_MODELS / f"policy-model-final.pt")
torch.save(target_DQN.state_dict(), PATH_MODELS / f"target-model-final.pt")
print("Complete")
