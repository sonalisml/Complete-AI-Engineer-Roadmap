def calculate_grade(average):
    if average >= 90:
        return "A+"
    elif average >= 80:
        return "A"
    elif average >= 70:
        return "B"
    elif average >= 60:
        return "C"
    elif average >= 40:
        return "D"
    else:
        return "F"


def calculate_result(average):
    if average >= 40:
        return "PASS"
    else:
        return "FAIL"


def get_student_marks(subjects):

    marks = []

    for subject in subjects:
        mark = float(input(f"Enter marks for {subject}: "))
        marks.append(mark)

    return marks


def display_result(name, subjects, marks, average,
                   highest_mark, lowest_mark, grade, result):

    print("\n========== STUDENT RESULT ==========")

    print("Name:", name)

    print("\n----- SUBJECT-WISE MARKS -----")

    for subject, mark in zip(subjects, marks):
        print(f"{subject}: {mark}")

    print("\nAverage:", round(average, 2))
    print("Highest Mark:", highest_mark)
    print("Lowest Mark:", lowest_mark)
    print("Grade:", grade)
    print("Result:", result)


# -------- MAIN PROGRAM --------

name = input("Enter student name: ")

subjects = ["Python", "Mathematics", "Machine Learning"]

marks = get_student_marks(subjects)

average = sum(marks) / len(marks)

highest_mark = max(marks)
lowest_mark = min(marks)

grade = calculate_grade(average)

result = calculate_result(average)

display_result(
    name,
    subjects,
    marks,
    average,
    highest_mark,
    lowest_mark,
    grade,
    result
)