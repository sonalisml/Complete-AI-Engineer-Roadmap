expenses = {} # create an dictionary

while True:
    print("expense tracker")
    print("1. add an expense")
    print("2. view the expenses")
    print("3. calculate total expense")
    print("4. Remove an expense category")
    print("5.Exit")

    choice = input("enter ur choice")
    if choice == "1":
        category = input("Enter expense category: ")
        amount = float(input("Enter expense amount: "))
        if category in expenses:
            expenses[category] = expenses[category]+amount
        else:
            expenses[category] = amount
        print("Expense added successfully!")
    elif choice == "2":
        if len(expenses)== 0:
             print("no expense aavailable")
        else:
           for category, amount in expenses.items():
                print(f"{category}:{amount}")
    elif choice == "3":
        total_expense = sum(expenses.values())
        print(f"Total Expense: ₹{total_expense}")
    elif choice == "4":
         category = input("enter an category")
         if category in expenses:
            del expenses[category]
    elif choice == "5":
         print("Good Bye")
         break
    else:
        print("Invalid choice! Please enter a number between 1 and 5.")
