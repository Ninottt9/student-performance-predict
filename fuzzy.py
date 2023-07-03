import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyControls:
    def __init__(self):

        self.antecedent_universes = {
            'study_time': np.arange(1, 11, 1),
            'absences': np.arange(0, 94, 1),
            'health': np.arange(1, 6, 1),
        }

        self.grade_range = np.arange(0, 21, 1)

        self.study_time = ctrl.Antecedent(self.antecedent_universes['study_time'], 'study_time')
        self.absences = ctrl.Antecedent(self.antecedent_universes['absences'], 'absences')
        self.health = ctrl.Antecedent(self.antecedent_universes['health'], 'health')
        self.grade = ctrl.Consequent(self.grade_range, 'grade')
        self.grade.defuzzify_method = 'lom'

    def init_control_system(self):

        self.rule1 = ctrl.Rule(self.study_time['low'] & self.absences['low'] & self.health['low'], self.grade['low'])
        self.rule2 = ctrl.Rule(self.study_time['low'] & self.absences['low'] & self.health['high'], self.grade['low'])
        self.rule3 = ctrl.Rule(self.study_time['low'] & self.absences['medium'] & self.health['high'], self.grade['low'])
        self.rule4 = ctrl.Rule(self.study_time['medium'] & self.absences['medium'] & self.health['medium'], self.grade['medium'])
        self.rule5 = ctrl.Rule(self.study_time['medium'] & self.absences['low'] & self.health['high'], self.grade['medium'])
        self.rule6 = ctrl.Rule(self.study_time['medium'] & self.absences['low'] & self.health['medium'], self.grade['medium'])
        self.rule7 = ctrl.Rule(self.study_time['high'] & self.absences['low'] & self.health['high'], self.grade['high'])
        self.rule8 = ctrl.Rule(self.study_time['high'] & self.absences['low'] & self.health['medium'], self.grade['high'])
        self.rule9 = ctrl.Rule(self.study_time['high'] & self.absences['medium'] & self.health['medium'], self.grade['medium'])
        
        

        self.control_system = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4, self.rule5, self.rule6, self.rule7, self.rule8, self.rule9])
        self.control_simulation = ctrl.ControlSystemSimulation(self.control_system)
