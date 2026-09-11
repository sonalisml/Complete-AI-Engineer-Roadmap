subjects = ["python", "ml", "mathematics"]
marks = []

for subject in subjects:
    mark = float(input(f"Enter the marks for {subject}:"))
    marks.append(mark)


for subject, mark in zip(subjects, marks):
    print(f"{subject}:{marks}")

average = sum(marks) / len(marks)
highest_mark = max(marks)
lowest_mark = min(marks)

def calculate_grade(average):

    if average >= 90:
        return "A+"
    elif average >= 80:
        return "A"
    else:
        return "F"

grade = calculate_grade(average)