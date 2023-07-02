import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

from fuzzy import FuzzyControls
from Solution import Solution

def train(attributes_train, grades_train):
    start_time = time.time()

    # initialize the fuzzy control system
    fzCtrl = FuzzyControls()

    # define a function to perform tournament selection
    def perform_selection(population, tournament_size):
        selected_indices = []

        for _ in range(len(population)):
            candidates = random.choices(range(len(population)), k=tournament_size)
            winner = max(candidates, key=lambda x: population[x].fitness)
            selected_indices.append(winner)

        return [population[i] for i in selected_indices]

    # Define the genetic algorithm parameters
    population_size = 50
    num_generations = 50
    mutation_rate = 0.2
    tournament_size = 5

    # Initialize the population
    population = []
    for _ in range(population_size):
        individual = Solution(fzCtrl)
        population.append(individual)

    before_training = time.time()
    print(f"--- TIME TO READ THE DATA AND INITIALIZE THE POPULATION {(before_training - start_time)} seconds ---")

    # Initialize lists to store average fitness values
    average_fitness_population = []
    average_fitness_best = []

    # Run the genetic algorithm
    for generation in range(num_generations):
        before_itaration = time.time()

        fitness_values = []
        for individual in population:
            individual.evaluate_fitness(attributes_train, grades_train)
            fitness_values.append(individual.fitness)
        
        best_individual = max(population, key=lambda x: x.fitness)

        # Calculate average fitness values
        average_fitness_population.append(np.mean(fitness_values))
        average_fitness_best.append(best_individual.fitness)
        
        # Print the best individual in each generation
        print("Generation:", generation+1)
        # print("Best Individual:\nAntecedent Ranges:\n", best_individual.antecedent_ranges)
        # print("Consequesnt ranges:\n", best_individual.consequent_range)
        print("Best Fitness:", best_individual.fitness)
        
        # Perform selection
        selected_population = perform_selection(population, tournament_size)
        
        # Create the next generation through crossover and mutation
        offspring_population = []
        while len(offspring_population) < population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            child1, child2 = parent1.crossover(parent2)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            offspring_population.append(child1)
            offspring_population.append(child2)
        
        population = offspring_population

        iteration_time = time.time() - before_itaration
        print("--- ITERATION %.2f seconds ---\n" % iteration_time)

    print(f"--- TOTAL time training took {((time.time() - before_training)/60):.2f} minutes ---")

    print("\nBEST")
    # # Evaluate the best individual
    for individual in population:
        individual.evaluate_fitness(attributes_train, grades_train)
    best_individual = max(population, key=lambda x: x.fitness)
    best_individual.update_fuzzy_control_system()
    print("Best individual fitness:", best_individual.fitness)
    print("Best Individual:\nAntecedent Ranges:\n", best_individual.antecedent_ranges)
    print("Consequesnt ranges:\n", best_individual.consequent_range)

    print(f" --- TOTAL time algorithm took {((time.time() - start_time)/60):.2f} minutes ---")

    for antecedant in best_individual.antecedent_ranges:
        getattr(fzCtrl, antecedant).view()
    plt.show()

    generations = range(num_generations)
    plt.plot(generations, average_fitness_population, label='Average Fitness (Population)')
    plt.plot(generations, average_fitness_best, label='Average Fitness (Best Individual)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

    return best_individual