from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

# Hyperparameters
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 30000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1

# Create Network using Keras API
def createNetwork():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(80, 80, 4)),
        tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(ACTIONS)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Train Network
def trainNetwork(model):
    game_state = game.GameState()
    D = deque()
    epsilon = INITIAL_EPSILON
    t = 0

    # Get initial state
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, _, _ = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    x_t = np.expand_dims(x_t, axis=2)
    s_t = np.repeat(x_t, 4, axis=2)
    
    while True:
        # Choose action
        if random.random() <= epsilon:
            action_index = random.randrange(ACTIONS)
        else:
            q_values = model.predict(np.expand_dims(s_t, axis=0), verbose=0)
            action_index = np.argmax(q_values)
        
        a_t = np.zeros([ACTIONS])
        a_t[action_index] = 1
        
        # Take action
        x_t1, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1, (80, 80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.expand_dims(x_t1, axis=2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        
        # Store transition
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        # Training
        if t > OBSERVE and len(D) > BATCH:
            minibatch = random.sample(D, BATCH)
            s_batch, a_batch, r_batch, s1_batch, terminal_batch = zip(*minibatch)
            q_targets = model.predict(np.array(s_batch), verbose=0)
            q_next = model.predict(np.array(s1_batch), verbose=0)
            
            for i in range(BATCH):
                if terminal_batch[i]:
                    q_targets[i][np.argmax(a_batch[i])] = r_batch[i]
                else:
                    q_targets[i][np.argmax(a_batch[i])] = r_batch[i] + GAMMA * np.max(q_next[i])
            
            model.fit(np.array(s_batch), q_targets, epochs=1, verbose=0)
        
        s_t = s_t1
        t += 1
        
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        
        print(f"TIMESTEP {t} / EPSILON {epsilon:.6f} / ACTION {action_index} / REWARD {r_t}")

# Start game and training
def main():
    model = createNetwork()
    trainNetwork(model)

if __name__ == "__main__":
    main()
