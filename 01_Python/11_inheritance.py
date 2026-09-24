class vehicle:

    def __init__(self, name):
        self.name = name

    def display_name(self):
        print(self.name)

#Child class   
class car(vehicle):

    def __init__(self, name, speed):
        super().__init__(name) #It will call the parents class constructor.
        self.speed = speed

    def display_speed(self):
         print(f"Speed: {self.speed} km/h")

#Create an object
car1 = car("Tesla", 200)
car1.display_name()
car1.display_speed()