
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from DBConnection import DBConnection

def Dinner_accuracy():
    food_dataset = pd.read_csv('food3_2.csv',encoding="ISO-8859-1")
    #print(len(food_dataset))
    #print(food_dataset.isnull().sum().sum())
    #print(food_dataset.isnull().values.any())
    rec_food = food_dataset

    Dinnerdata = food_dataset['Dinner']

    DinnerdataNumpy = Dinnerdata.to_numpy()

    Food_itemsdata = food_dataset['Food_items']

    Dinnerfoodseparated = []
    DinnerfoodseparatedID = []

    for i in range(len(Dinnerdata)):
        if DinnerdataNumpy[i] == 1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)


    DinnerfoodseparatedIDdata = food_dataset.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    val = list(np.arange(8, 19))
    nutritions = [0] + val
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.iloc[nutritions]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T




    # converting into numpy array
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()


    ## K-Means Based  Dinner Food
    Datacalorie = DinnerfoodseparatedIDdata[0:, 1:len(DinnerfoodseparatedIDdata)]

    #print(type(Datacalorie))
    #print("len=",len(DinnerfoodseparatedIDdata))
    X = np.array(Datacalorie)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

    XValu = np.arange(0, len(kmeans.labels_))

    # retrieving the labels for dinner food
    dnrlbl = kmeans.labels_


    ## Reading of the Dataet
    datafin_nutr = pd.read_csv('nutrition_distriution.csv')
    ## train set
    datafin_nutr = datafin_nutr.T
    #print("datafin_nutr=",datafin_nutr)

    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]

    weightlosscat = datafin_nutr.iloc[[1, 2, 7, 8]]
    #print("wlcat=", weightlosscat)
    weightlosscat = weightlosscat.T

    # Converting numpy array
    weightlosscatDdata = weightlosscat.to_numpy()

    weightlosscat = weightlosscatDdata[0:, 0:len(weightlosscatDdata)]

    #print("wlcatdata=", weightlosscat)

    weightlossfin = np.zeros((len(weightlosscat) * 5, 4), dtype=np.float32)
    #print("weightlossfin=",weightlossfin)
    #print("weightlossfinlen=", len(weightlossfin))

    t = 0
    r = 0
    s = 0
    yt = []
    yr = []
    ys = []
    for zz in range(5):
        #print("zz=", zz)
        for jj in range(len(weightlosscat)):
            #print("jj=",jj)
            valloc = list(weightlosscat[jj])
            #print("nval=", np.array(valloc))
            weightlossfin[t] = np.array(valloc)
            #print("brklbl=", lnchlbl[jj])
            yt.append(dnrlbl[jj])
            t += 1


    X_train = weightlossfin  # Features
    y_train = yt  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=40)



    clf = RandomForestClassifier(n_estimators=300, max_depth=2)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_dnr = accuracy_score(y_test, y_pred) * 100

    print("Dinner_accuracy=",accuracy_dnr)

    return accuracy_dnr



#Dinner_accuracy()








