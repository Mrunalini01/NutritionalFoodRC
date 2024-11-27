
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def  Lunch_accuracy():
    food_dataset = pd.read_csv('food3_2.csv',encoding="ISO-8859-1")

    #print(food_dataset.isnull().values.any())
    rec_food = food_dataset
    Lunchdata = food_dataset['Lunch']
    LunchdataNumpy = Lunchdata.to_numpy()

    Food_itemsdata = food_dataset['Food_items']

    #check_for_nan = food_dataset['Food_items'].isnull().values.any()
    #print(check_for_nan)


    Lunchfoodseparated = []
    LunchfoodseparatedID = []

    for i in range(len(Lunchdata)):
        if LunchdataNumpy[i] == 1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)

    # retrieving Lunch data rows by loc method |
    LunchfoodseparatedIDdata = food_dataset.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    val = list(np.arange(8, 19))
    nutritions = [0] + val
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.iloc[nutritions]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T




    # converting into numpy array
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    #print(LunchfoodseparatedIDdata)


    #print("len=", len(LunchfoodseparatedIDdata))
    ## K-Means Based  lunch Food
    Datacalorie = LunchfoodseparatedIDdata[0:, 1:len(LunchfoodseparatedIDdata)]

    X = np.array(Datacalorie)

    kmeans = KMeans(n_clusters=3).fit(X)

    XValu = np.arange(0, len(kmeans.labels_))

    # retrieving the labels for lunch food
    lnchlbl = kmeans.labels_
    #print("lunchlbl=", lnchlbl)


    ## Reading of the Dataet
    datafin_nutr = pd.read_csv('nutrition_distriution.csv')
    ## train set
    datafin_nutr = datafin_nutr.T
    #print("datafin_nutr=",datafin_nutr)

    weightlosscat = datafin_nutr.iloc[[1, 2, 7, 8]]
    #print("wlcat=", weightlosscat)
    weightlosscat = weightlosscat.T

    # Converting numpy array
    weightlosscatDdata = weightlosscat.to_numpy()

    weightlosscat = weightlosscatDdata[0:, 0:len(weightlosscatDdata)]

    #print("wlcatdata=", weightlosscat)

    weightlossfin = np.zeros((len(weightlosscat) * 5, 4), dtype=np.float32)

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
            yt.append(lnchlbl[jj])
            t += 1



    X_train = weightlossfin  # Features
    y_train = yt  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=40)



    clf = RandomForestClassifier(n_estimators=300,max_depth=2)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_lnch = accuracy_score(y_test, y_pred) * 100
    print("Lunch_accuracy=",accuracy_lnch)
    return accuracy_lnch


#Lunch_accuracy()









