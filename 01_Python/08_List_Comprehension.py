numbers = [10, 20, 30, 40]

#write it in one line to generate square or even number
squares = [number **2 for number in numbers]
print(squares)

even_number = [number for number in numbers if number%2==0]
print(even_number)


#write a list comprehension to create a list containing the squares of only even numbers.
sq_evennum = [number**2 for number in even_number]
print(sq_evennum)
+