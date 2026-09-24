#WRITE
file = open("data.txt", "w")
file.write("sonali\n")
file.write("rahul\n")
file.close()
print("file created successfully")

#READ
file = open("data.txt", "r")
content = file.read()
print(content)
file.close()

#USE WITH
with open("data.txt", "r")
     content = file.read()
     print(content)

with open("data.txt", "a") as file:
     file.write("anita\n")
     print(content)

with open("data.txt", "r") as file:
     for line in line:
         print(line.strip())