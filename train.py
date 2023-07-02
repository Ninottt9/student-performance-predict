import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fuzzy import FuzzyControls
from Solution import Solution

# read the data
# data = np.genfromtxt('student/student-por.csv', delimiter=';', skip_header=1)
data = pd.read_csv('student/student-por.csv', delimiter=';').values

# take 14th column as study_time, 30th column as absences, 29th columnd as health, 27 as alcohol
data = data[:, [13, 29, 28, 26, 31]]

# print(data.shape)
# print(data)

# split into attributes and grades
attributes = data[:, :-1]
grades = data[:, -1]

# print(attributes)
# print(grades)
# split into training and test sets
attributes_train, attributes_test, grades_train, grades_test = train_test_split(attributes, grades, test_size=0.2, random_state=42)

def perform_selection(population, tournament_size):
    selected_indices = []

    for _ in range(len(population)):
        candidates = random.choices(range(len(population)), k=tournament_size)
        winner = max(candidates, key=lambda x: population[x].fitness)
        selected_indices.append(winner)

    return [population[i] for i in selected_indices]

fzCtrl = FuzzyControls()

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

# Run the genetic algorithm
for generation in range(num_generations):
    for individual in population:
        individual.evaluate_fitness(attributes_train, grades_train)
    
    best_individual = max(population, key=lambda x: x.fitness)
    
    # Print the best individual in each generation
    print("Generation:", generation, "Best Individual:", best_individual.antecedent_universes, "Best Fitness:", best_individual.fitness)
    
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

print("BEEST")
# Evaluate the best individual
best_individual = max(population, key=lambda x: x.fitness)
print("Best individual fitness:", best_individual.fitness)

# Testing phase
# Apply the best individual to the test set
predicted_grades = []
for attributes in attributes_test:
    predicted_grade = best_individual.predict_grade(attributes)  # Predict grade using fuzzy logic
    predicted_grades.append(predicted_grade)

print("Predicted Grades (Test):", grades_test)
print("Actual Grades (Test):", grades_test)

# Evaluate performance on the test set
mse = np.mean((grades_test - predicted_grades) ** 2)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE) on test set:", rmse)