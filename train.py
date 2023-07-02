import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

from fuzzy import FuzzyControls
from Solution import Solution

start_time = time.time()

# read the data
data = pd.read_csv('student/student-mat.csv', delimiter=';').values

# take 14th column as study_time, 30th column as absences, 29th columnd as health, 27 as alcohol, 32 as final grades
data = data[:, [13, 29, 28, 26, 31]]

# split into attributes and grades
attributes = data[:, :-1]
grades = data[:, -1]

# split into training and test sets
attributes_train, attributes_test, grades_train, grades_test = train_test_split(attributes, grades, test_size=0.2, random_state=42)


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
num_generations = 10
mutation_rate = 0.2
tournament_size = 5

# Initialize the population
population = []
for _ in range(population_size):
    individual = Solution(fzCtrl)
    population.append(individual)

before_training = time.time() - start_time
print("--- BEFORE ALGORITHM TRAINING %s seconds ---" % before_training)

# Run the genetic algorithm
for generation in range(num_generations):
    before_itaration = time.time()

    for individual in population:
        individual.evaluate_fitness(attributes_train, grades_train)
    
    best_individual = max(population, key=lambda x: x.fitness)
    
    # Print the best individual in each generation
    print("Generation:", generation+1, "Best Fitness:", best_individual.fitness)
    
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
    print("--- ITERATION %.2f seconds ---" % iteration_time)

print("--- TOTAL time training took %.2f seconds ---" % (time.time() - before_training))

print("\nBEST")
# Evaluate the best individual
best_individual = max(population, key=lambda x: x.fitness)
print("Best individual fitness:", best_individual.fitness)

# Testing phase
# Apply the best individual to the test set
predicted_grades = []
for attributes in attributes_test:
    predicted_grade = best_individual.predict_grade(attributes)  # Predict grade using fuzzy logic
    predicted_grades.append(predicted_grade)

print("Predicted Grades (Test):", predicted_grades)
print("Actual Grades (Test):", grades_test)

# Evaluate performance on the test set
mse = np.mean((grades_test - predicted_grades) ** 2)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE) on test set:", rmse)

print(" --- TOTAL time algorithm took %s seconds ---" % (time.time() - start_time))