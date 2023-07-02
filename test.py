import numpy as np

def test(best_individual, attributes_test, grades_test):
    # Testing phase
    # Apply the best individual to the test set
    predicted_grades = []
    for attributes in attributes_test:
        predicted_grade = best_individual.predict_grade(attributes)  # Predict grade using fuzzy logic
        predicted_grades.append(predicted_grade)

    # Write the results to a file
    with open('output.txt', 'w') as file:
        file.write("Predicted Grades (Test)\tActual Grades (Test)\n")
        for predicted_grade, actual_grade in zip(predicted_grades, grades_test):
            file.write(f"{predicted_grade}\t{actual_grade}\n")

    # Evaluate performance on the test set
    mse = np.mean((grades_test - predicted_grades) ** 2)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE) on test set:", rmse)