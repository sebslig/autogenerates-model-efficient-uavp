import json
import random
from collections import namedtuple

# Define the search space operations
Operation = namedtuple('Operation', ['name', 'params'])

OP_TYPES = [
    Operation('conv_3x3', {'out_channels': [16, 32, 64]}),
    Operation('conv_5x5', {'out_channels': [16, 32, 64]}),
    Operation('max_pool_2x2', {}),
    Operation('avg_pool_2x2', {})
]

NUM_LAYERS = 5 # Number of operations in a cell
POPULATION_SIZE = 10
NUM_GENERATIONS = 5

def create_random_architecture():
    """Generates a random sequence of operations."""
    architecture = []
    for _ in range(NUM_LAYERS):
        op = random.choice(OP_TYPES)
        # For simplicity, if op has params, pick a random one, otherwise empty dict
        selected_params = {k: random.choice(v) for k, v in op.params.items()} if op.params else {}
        architecture.append({'op': op.name, 'params': selected_params})
    return architecture

def evaluate_architecture(architecture):
    """Placeholder for actual model evaluation. Returns a dummy 'accuracy'."""
    # In a real scenario, this would build and train a model, then return its performance.
    # For this example, we'll assign a random score.
    return random.uniform(0.1, 0.9) # Simulate accuracy

def crossover(parent1, parent2):
    """Performs one-point crossover."""
    crossover_point = random.randint(1, NUM_LAYERS - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(architecture, mutation_rate=0.2):
    """Randomly mutates an operation in the architecture."""
    if random.random() < mutation_rate:
        layer_idx = random.randint(0, NUM_LAYERS - 1)
        new_op = random.choice(OP_TYPES)
        selected_params = {k: random.choice(v) for k, v in new_op.params.items()} if new_op.params else {}
        architecture[layer_idx] = {'op': new_op.name, 'params': selected_params}
    return architecture

def genetic_search():
    population = [create_random_architecture() for _ in range(POPULATION_SIZE)]
    best_architecture = None
    best_score = -1

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}")
        scores = [(arch, evaluate_architecture(arch)) for arch in population]
        scores.sort(key=lambda x: x[1], reverse=True)

        current_best_arch, current_best_score = scores[0]
        if current_best_score > best_score:
            best_score = current_best_score
            best_architecture = current_best_arch

        print(f"  Best score in this generation: {current_best_score:.4f}")

        # Select top half for reproduction
        parents = [arch for arch, _ in scores[:POPULATION_SIZE // 2]]
        next_population = parents[:]

        # Generate new offspring
        while len(next_population) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            child1, child2 = crossover(p1, p2)
            next_population.append(mutate(child1))
            if len(next_population) < POPULATION_SIZE:
                next_population.append(mutate(child2))
        population = next_population

    print(f"\nSearch complete. Best score found: {best_score:.4f}")
    return best_architecture, best_score

if __name__ == '__main____':
    import os
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    best_arch, best_score = genetic_search()
    print("\nBest Architecture:", json.dumps(best_arch, indent=2))

    with open(os.path.join(output_dir, 'best_architecture.json'), 'w') as f:
        json.dump({'architecture': best_arch, 'score': best_score}, f, indent=2)
