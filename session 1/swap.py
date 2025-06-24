# Swap values without using a third variable
x = int(input("Enter value of x: "))
y = int(input("Enter value of y: "))

print(f"Before swapping: x = {x}, y = {y}")
x, y = y, x
print(f"After swapping: x = {x}, y = {y}")
