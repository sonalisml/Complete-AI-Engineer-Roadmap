#create an empty task
tasks = []

wtd = ["learnpython", "learnpytorch", "learn ml"]

for task in wtd:
    tasks.append(task)
    print(task)

while True:
    print("To do list Menu")
    print("1.add task")
    print("2.view task")
    print("3.exit")

    choice = input("enter the choice")

    if choice == "3":
        print("Goodbye")
        break
    elif choice == "1":
        task = input("Enter a new task: ")
        tasks.append(task)
        print("Task added successfully!")
    elif choice == "2":
        print("\n----- YOUR TASKS -----")
        for index, task in enumerate(tasks, start=1):
            print(f"{index}.{task}")