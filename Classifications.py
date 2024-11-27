from Breakfast_accuracy import Breakfast_accuracy

from Lunch_accuracy import Lunch_accuracy

from Dinner_accuracy import Dinner_accuracy

from barcharts import view

def evaluation():


    bf_accuracy=Breakfast_accuracy()

    lnh_accuracy=Lunch_accuracy()

    dnr_accuracy=Dinner_accuracy()


    list=[]
    list.clear()
    list.append(bf_accuracy)
    list.append(lnh_accuracy)
    list.append(dnr_accuracy)
    view(list)

evaluation()


