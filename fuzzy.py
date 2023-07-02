import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyControls:
    def __init__(self): 
        self.study_time_range = np.arange(1, 11, 1)
        self.absences_range = np.arange(0, 94, 1)
        self.health_range = np.arange(1, 6, 1)
        self.alcohol_range = np.arange(1, 6, 1)
        self.grade_range = np.arange(0, 21, 1)

        self.study_time = ctrl.Antecedent(self.study_time_range, 'study_time')
        self.absences = ctrl.Antecedent(self.absences_range, 'absences')
        self.health = ctrl.Antecedent(self.health_range, 'health')
        self.alcohol = ctrl.Antecedent(self.alcohol_range, 'alcohol')
        self.grade = ctrl.Consequent(self.grade_range, 'grade')
        self.grade.defuzzify_method = 'lom'

        # Define the fuzzy membership functions and rules here

    def init_control_system(self):
        self.rule1 = ctrl.Rule(self.study_time['low'] & self.absences['low'] & self.health['low'] & self.alcohol['low'], self.grade['low'])
        self.rule2 = ctrl.Rule(self.study_time['medium'] & self.absences['medium'] & self.health['medium'] & self.alcohol['medium'], self.grade['medium'])
        self.rule3 = ctrl.Rule(self.study_time['high'] & self.absences['low'] & self.health['high'] & self.alcohol['low'], self.grade['high'])
        self.rule4 = ctrl.Rule(self.study_time['medium'] & self.absences['low'] & self.health['high'] & self.alcohol['low'], self.grade['medium'])
        self.rule5 = ctrl.Rule(self.study_time['high'] & self.absences['low'] & self.health['high'] & self.alcohol['low'], self.grade['high'])
        self.rule6 = ctrl.Rule(self.study_time['low'] & self.absences['low'] & self.health['high'] & self.alcohol['low'], self.grade['medium'])
        self.rule7 = ctrl.Rule(self.study_time['medium'] & self.absences['medium'] & self.health['medium'] & self.alcohol['high'], self.grade['low'])
        self.rule8 = ctrl.Rule(self.study_time['low'] & self.absences['medium'] & self.health['high'] & self.alcohol['high'], self.grade['low'])

        self.control_system = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4, self.rule5, self.rule6, self.rule7, self.rule8])
        self.control_simulation = ctrl.ControlSystemSimulation(self.control_system)
