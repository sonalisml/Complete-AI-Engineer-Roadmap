class student: #CLASS CREATION
    def __init__(self,name,marks):   #Initialize the object
        self.name = name
        self.marks = marks
    def display(self):
        print(self.name)
        print(self.marks)
    def check_result(self):
        if self.marks >40:
            print(f"{self.name}:Pass")
        else:
            print("Fail")

#CREATE  OBJECT
student1 = student("sonali", 55)
student2 = student("rahul", 25)

#CALL THE METHOD VIA OBJECT
student1.display()
student2.display()
student1.check_result()
student2.check_result()