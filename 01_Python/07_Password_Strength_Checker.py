password = input("Enter your password")
print("Your password is ", password)
password_length = len(password)
print(password_length)

# Initially assume nothing is found
has_lower = False
has_upper = False
has_digit = False
has_special = False

for character in password:
    if character.islower():
         has_lower = True
    elif character.isupper():
         has_upper = True
    elif character.isdigit():
         has_digit = True
    else:
         has_special = True


#length, uppercase, lower case, digit
if password_length <=6:
    print("Weak password")
elif len(password) >6 and has_lower and has_upper and has_digit:
    print("strong passowrd")
else:
    print("medium password")