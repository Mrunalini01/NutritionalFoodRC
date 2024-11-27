from Breakfast2 import Diet_Control_Breakfast
from Lunch2 import Diet_Control_Lunch
from Dinner2 import Diet_Control_Dinner
age=int(input("Enter Age:"))

gender=int(input("Select Gender(1-Male,2-Female): "))

height=int(input("Enter Height(cm) :"))

weight=int(input("Enter Weight(kg) :"))

cuisine=input("Select Cuisine(South Indian Recipes,North Indian Recipes): ")

catgry=input("Select Category(Vegetarian,Non Vegeterian,Eggetarian): ")

sugar=int(input("Do You have a Sugar? (Yes-1,No-0): "))

bp=int(input("Do You have a BP? (Yes-1,No-0): "))


diet_catg=int(input("Select Diet Category? (WeightLoss-1,Weightgain-2,Health-3): "))


if diet_catg==1:
    wl=[1, 2, 7, 8]
    cols=4
    Diet_Control_Breakfast(age,gender,height,weight,cuisine,catgry,sugar,bp,wl,cols)
    Diet_Control_Lunch(age,gender, height, weight, cuisine, catgry, sugar, bp, wl, cols)
    Diet_Control_Dinner(age, gender, height, weight, cuisine, catgry, sugar, bp, wl, cols)
elif diet_catg==2:
    wl = [0,1,2,3,4,7,9,10]
    cols = 8
    Diet_Control_Breakfast(age, gender, height, weight, cuisine, catgry, sugar, bp, wl, cols)
    Diet_Control_Lunch(age, gender, height, weight, cuisine, catgry, sugar, bp, wl, cols)
    Diet_Control_Dinner(age, gender, height, weight, cuisine, catgry, sugar, bp, wl, cols)

else:
    wl = [1,2,3,4,6,7,9]
    cols = 7
    Diet_Control_Breakfast(age, gender, height, weight, cuisine, catgry, sugar, bp, wl, cols)
    Diet_Control_Lunch(age, gender, height, weight, cuisine, catgry, sugar, bp, wl, cols)
    Diet_Control_Dinner(age, gender, height, weight, cuisine, catgry, sugar, bp, wl, cols)


