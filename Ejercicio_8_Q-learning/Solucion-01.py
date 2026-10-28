import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. Configuración del entorno y parámetros
seed = 7
np.random.seed(seed)
alpha = 0.5
gamma = 1.0
epsilon = 0.1
episodes = 500
max_steps = 200

print("Cliff Walking - Q-learning")
print(f"alpha={alpha}, gamma={gamma}, epsilon={epsilon}, episodes={episodes}, seed={seed}\n")

# 2. Inicialización de la Tabla Q
# Estados: 4 filas x 12 columnas. Acciones: 4 (0:N, 1:E, 2:S, 3:W)
q_table = np.zeros((4, 12, 4))
actions_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
action_symbols = ['N', 'E', 'S', 'W'] # Letras 

def step(state, action):
    r, c = state
    dr, dc = actions_map[action]
    next_r, next_c = r + dr, c + dc
    
    # Límites del mapa
    if next_r < 0 or next_r >= 4 or next_c < 0 or next_c >= 12:
        next_r, next_c = r, c 
        
    # Acantilado (The Cliff)
    if next_r == 3 and 1 <= next_c <= 10:
        return (3, 0), -100, False 
        
    # Meta (G)
    if next_r == 3 and next_c == 11:
        return (3, 11), -1, True 
        
    # Transición ordinaria
    return (next_r, next_c), -1, False

def get_action(state, q_table, eps):
    if np.random.rand() < eps:
        return np.random.randint(4)
    else:
        return np.argmax(q_table[state[0], state[1]])

# 3. Entrenamiento (Q-Learning)
rewards_per_episode = []

for ep in range(episodes):
    state = (3, 0)
    total_reward = 0
    
    for _ in range(max_steps):
        action = get_action(state, q_table, epsilon)
        next_state, reward, done = step(state, action)
        
        total_reward += reward
        
        # Backup Q-learning
        if done:
            td_target = reward 
        else:
            td_target = reward + gamma * np.max(q_table[next_state[0], next_state[1]])
            
        td_error = td_target - q_table[state[0], state[1], action]
        q_table[state[0], state[1], action] += alpha * td_error
        
        state = next_state
        if done:
            break
            
    rewards_per_episode.append(total_reward)

# 4. Evaluación Greedy
state = (3, 0)
greedy_path = [state]
greedy_reward = 0
done = False

for _ in range(50):
    action = np.argmax(q_table[state[0], state[1]])
    next_state, reward, done = step(state, action)
    greedy_reward += reward
    greedy_path.append(next_state)
    state = next_state
    if done:
        break

# 5. Salida de la consola.
policy_grid = np.full((4, 12), '·', dtype=object)
for r in range(4):
    for c in range(12):
        if r == 3 and 1 <= c <= 10:
            policy_grid[r, c] = 'C'
        elif r == 3 and c == 11:
            policy_grid[r, c] = 'G'
        elif r == 3 and c == 0:
            policy_grid[r, c] = '@' # Marcador de inicio
        else:
            best_a = np.argmax(q_table[r, c])
            policy_grid[r, c] = action_symbols[best_a]

print("Greedy policy (@=start, G=goal, C=cliff; other cells show N/E/S/W):")
for row in policy_grid:
    print("  " + " ".join(row))

print("\nCamino greedy:")
path_str = " -> ".join([f"({r}, {c})" for r, c in greedy_path])
print(path_str)
print(f"Pasos: {len(greedy_path) - 1}")
print(f"Greedy return (epsilon=0): {greedy_reward}")

mean_training_return = np.mean(rewards_per_episode)
print(f"Mean training return (epsilon=0.1): {mean_training_return:.2f}")

# 6. Guardar datos en CSV 
csv_filename = os.path.join(os.getcwd(), 'episode_returns.csv')
df = pd.DataFrame({'Reward': rewards_per_episode})
df.to_csv(csv_filename, index=False)
print(f"Recompensas por episodio guardadas en: {csv_filename}")

# 7. Generar Gráfica
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.3, label='Recompensa por episodio (Cruda)', color='blue')
window = 10
moving_avg = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
plt.plot(np.arange(window-1, len(rewards_per_episode)), moving_avg, color='red', label=f'Media Móvil (n={window})')
plt.ylim(-150, 0)
plt.axhline(-13, color='green', linestyle='--', label='Retorno Óptimo (-13)')
plt.title('Entrenamiento de Q-learning en Cliff Walking')
plt.xlabel('Episodios')
plt.ylabel('Recompensa (Retorno)')
plt.legend()
plt.grid(True)
plt.savefig('cliff_walking_qlearning.png')