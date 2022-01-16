import multiprocessing
import os
import pickle
import gym
import neat as neat
import numpy as np

runs_per_net = 2


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []
    for runs in range(runs_per_net):
        env = gym.make("BipedalWalker-v3")
        observation = env.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False
        while not done:
            action = net.activate(observation)  # predict action
            observation, reward, done, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    config_path = os.path.join('config_bipedal', )
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner_bipedal', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == '__main__':
    run()
