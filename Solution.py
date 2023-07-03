import random
import numpy as np
import skfuzzy as fuzz
import copy

class Solution:
    def __init__(self, fzCtrl):

        self.antecedent_ranges = {
            'study_time': [],
            'absences': [],
            'health': [],
        }
        self.consequent_range = []

        self.fitness = float('-inf')
        self.fzCtrl = fzCtrl

        self.initialize_ranges()

    def initialize_ranges(self):
        for antecedent in self.antecedent_ranges:
            self.antecedent_ranges[antecedent] = []
            for _ in range(3):
                min_val = np.random.choice(self.fzCtrl.antecedent_universes[antecedent])
                mid_val = np.random.choice(self.fzCtrl.antecedent_universes[antecedent])
                max_val = np.random.choice(self.fzCtrl.antecedent_universes[antecedent])
                sorted_vals = np.sort([min_val, mid_val, max_val])
                self.antecedent_ranges[antecedent].append(tuple(sorted_vals))

        self.consequent_range = []
        for _ in range(3):
            min_val = np.random.choice(self.fzCtrl.grade_range)
            max_val = np.random.choice(self.fzCtrl.grade_range)
            mid_val = np.random.choice(self.fzCtrl.grade_range)
            sorted_vals = np.sort([min_val, mid_val, max_val])
            self.consequent_range.append(tuple(sorted_vals))

        # Sort the ranges based on the maximum value
        self.antecedent_ranges = {key: sorted(value, key=lambda x: x[2]) for key, value in self.antecedent_ranges.items()}
        self.consequent_range.sort(key=lambda x: x[2])

    def evaluate_fitness(self, attributes, grades):
        self.update_fuzzy_control_system()

        predicted_grades = []
        for attribute in attributes:
            predicted_grade = self.predict_grade(attribute)
            predicted_grades.append(predicted_grade)

        self.fitness = -np.mean((grades - predicted_grades) ** 2)

    def update_fuzzy_control_system(self):
        for antecedent in self.antecedent_ranges:
            fz_antecedent = getattr(self.fzCtrl, antecedent)
            fz_antecedent['low'] = fuzz.trimf(self.fzCtrl.antecedent_universes[antecedent], self.antecedent_ranges[antecedent][0])
            fz_antecedent['medium'] = fuzz.trimf(self.fzCtrl.antecedent_universes[antecedent], self.antecedent_ranges[antecedent][1])
            fz_antecedent['high'] = fuzz.trimf(self.fzCtrl.antecedent_universes[antecedent], self.antecedent_ranges[antecedent][2])

        fz_grade = self.fzCtrl.grade
        fz_grade['low'] = fuzz.trimf(self.fzCtrl.grade_range, self.consequent_range[0])
        fz_grade['medium'] = fuzz.trimf(self.fzCtrl.grade_range, self.consequent_range[1])
        fz_grade['high'] = fuzz.trimf(self.fzCtrl.grade_range, self.consequent_range[2])

        self.fzCtrl.init_control_system()

    def predict_grade(self, attributes):
        self.fzCtrl.control_simulation.input['study_time'] = attributes[0]
        self.fzCtrl.control_simulation.input['absences'] = attributes[1]
        self.fzCtrl.control_simulation.input['health'] = attributes[2]
        self.fzCtrl.control_simulation.compute()

        predicted_grade = self.fzCtrl.control_simulation.output['grade']
        return predicted_grade

    def mutate(self, mutation_rate):
        for antecedent in self.antecedent_ranges:
            if random.random() < mutation_rate:
                self.antecedent_ranges[antecedent] = []
                for _ in range(3):
                    min_val = np.random.choice(self.fzCtrl.antecedent_universes[antecedent])
                    max_val = np.random.choice(self.fzCtrl.antecedent_universes[antecedent])
                    mid_val = np.random.choice(self.fzCtrl.antecedent_universes[antecedent])
                    sorted_vals = np.sort([min_val, mid_val, max_val])
                    self.antecedent_ranges[antecedent].append(tuple(sorted_vals))

        if random.random() < mutation_rate:
            self.consequent_range = []
            for _ in range(3):
                min_val = np.random.choice(self.fzCtrl.grade_range)
                max_val = np.random.choice(self.fzCtrl.grade_range)
                mid_val = np.random.choice(self.fzCtrl.grade_range)
                sorted_vals = np.sort([min_val, mid_val, max_val])
                self.consequent_range.append(tuple(sorted_vals))
            
        self.antecedent_ranges = {key: sorted(value, key=lambda x: x[2]) for key, value in self.antecedent_ranges.items()}
        self.consequent_range.sort(key=lambda x: x[2])

    def crossover(self, other):
        child1 = Solution(self.fzCtrl)
        child2 = Solution(self.fzCtrl)
        for antecedent in self.antecedent_ranges:
            crossover_point = random.randint(0, 2)
            child1.antecedent_ranges[antecedent][:crossover_point] = self.antecedent_ranges[antecedent][:crossover_point]
            child1.antecedent_ranges[antecedent][crossover_point:] = other.antecedent_ranges[antecedent][crossover_point:]
            child2.antecedent_ranges[antecedent][:crossover_point] = other.antecedent_ranges[antecedent][:crossover_point]
            child2.antecedent_ranges[antecedent][crossover_point:] = self.antecedent_ranges[antecedent][crossover_point:]

        crossover_point = random.randint(0, 2)
        child1.consequent_range[:crossover_point] = self.consequent_range[:crossover_point]
        child1.consequent_range[crossover_point:] = other.consequent_range[crossover_point:]
        child2.consequent_range[:crossover_point] = other.consequent_range[:crossover_point]
        child2.consequent_range[crossover_point:] = self.consequent_range[crossover_point:]

        return child1, child2
