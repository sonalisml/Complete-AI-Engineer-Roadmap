marks = [] # create an empty list 

#append the marks using marks.append method
#no variables are needed
marks.append(float(input("Enter the python marks")))
marks.append(float(input("enter the ml marks")))
marks.append(float(input("enter the math marks")))

average = sum(marks)/ len(marks)
highest_marks = max(marks)
lowest_marks= min(marks)
print(average)
print(average, highest_marks, lowest_marks)