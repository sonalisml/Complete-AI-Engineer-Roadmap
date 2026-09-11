name = input("Enter the nam eof the student")
python_marks = float(input("Enter the python marks"))
math_marks = float(input("Enter the math marks"))
ml_marks = float(input("Enter the ml marks"))

average = (python_marks+math_marks+ml_marks)/3

print("The average marks of the student:",name, "is")
print(average)

if average > 50:
    result = "Pass"
else:
    result = "Fail"

print("result", result)

if average >= 90:
    grade = "A+"
elif average >= 80:
    grade = "A"
elif average >= 70:
    grade = "B"
elif average >= 60:
    grade = "C"
elif average >= 40:
    grade = "D"
else:
    grade = "F"

print("\n----- STUDENT RESULT -----")

print("Name:", name)
print("Average:", average)
print("Grade:", grade)
print("Result:", result)

highest_mark = max(python_marks, math_marks, ml_marks)
lowest_mark = min(python_marks, math_marks, ml_marks)
print("highest mark", highest_mark)
print("Lowest mark", lowest_mark)