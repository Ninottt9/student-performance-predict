import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_fuzzy_set():
    study_time_range = np.arange(1, 11, 1)
    absences_range = np.arange(0, 94, 1)
    health_range = np.arange(1, 6, 1)
    alcohol_range = np.arange(1, 6, 1)

    study_time = ctrl.Antecedent(study_time_range, 'study_time')
    absences = ctrl.Antecedent(absences_range, 'absences')
    health = ctrl.Antecedent(health_range, 'health')
    alcohol = ctrl.Antecedent(alcohol_range, 'alcohol')
    grade = ctrl.Consequent(np.arange(0, 21, 1), 'grade')

    study_time['low'] = fuzz.trimf(study_time_range, [1, 1, 5])
    study_time['medium'] = fuzz.trimf(study_time_range, [2, 5, 8])
    study_time['high'] = fuzz.trimf(study_time_range, [5, 10, 10])

    absences['low'] = fuzz.trimf(absences_range, [0, 0, 30])
    absences['medium'] = fuzz.trimf(absences_range, [10, 30, 50])
    absences['high'] = fuzz.trimf(absences_range, [30, 70, 93])

    health['poor'] = fuzz.trimf(health_range, [1, 1, 3])
    health['average'] = fuzz.trimf(health_range, [1, 3, 5])
    health['good'] = fuzz.trimf(health_range, [3, 5, 5])

    alcohol['low'] = fuzz.trimf(alcohol_range, [1, 1, 3])
    alcohol['medium'] = fuzz.trimf(alcohol_range, [1, 3, 5])
    alcohol['high'] = fuzz.trimf(alcohol_range, [3, 5, 5])

    grade['low'] = fuzz.trimf(grade.universe, [0, 0, 10])
    grade['medium'] = fuzz.trimf(grade.universe, [5, 10, 15])
    grade['high'] = fuzz.trimf(grade.universe, [10, 20, 20])

    rule1 = ctrl.Rule(study_time['low'] & absences['low'] & health['poor'] & alcohol['low'], grade['low'])
    rule2 = ctrl.Rule(study_time['medium'] & absences['medium'] & health['average'] & alcohol['medium'], grade['medium'])
    rule3 = ctrl.Rule(study_time['high'] & absences['high'] & health['good'] & alcohol['high'], grade['high'])

    grading_system = ctrl.ControlSystem([rule1, rule2, rule3])
    grading = ctrl.ControlSystemSimulation(grading_system)

    #przyklad
    # grading.input['study_time'] = 7  # Przykładowa wartość: czas nauki = 7
    # grading.input['absences'] = 20  # Przykładowa wartość: nieobecności = 20
    # grading.input['health'] = 3  # Przykładowa wartość: zdrowie = 3
    # grading.input['alcohol'] = 4  # Przykładowa wartość: alkohol = 4
    # grading.compute()

    # grade_output = grading.output['grade']
    # print("The predicted grade is:", grade_output)

    return grading