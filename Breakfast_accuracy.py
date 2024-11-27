
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def Breakfast_accuracy():
    try:


        food_dataset = pd.read_csv('food3_2.csv', encoding="ISO-8859-1")

        rec_food = food_dataset
        Breakfastdata = food_dataset['Breakfast']
        BreakfastdataNumpy = Breakfastdata.to_numpy()
        Food_itemsdata = food_dataset['Food_items']

        breakfastfoodseparated = []
        breakfastfoodseparatedID = []

        for i in range(len(Breakfastdata)):
            if BreakfastdataNumpy[i] == 1:
                breakfastfoodseparated.append(Food_itemsdata[i])
                breakfastfoodseparatedID.append(i)

        # retrieving Breafast data rows by loc method
        breakfastfoodseparatedIDdata = food_dataset.iloc[breakfastfoodseparatedID]
        # print("bk=",breakfastfoodseparatedIDdata)
        # print("breakfastfoodseparatedID=",(breakfastfoodseparatedID))
        fddata = breakfastfoodseparatedIDdata
        breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
        val = list(np.arange(8, 19))
        nutritions = [0] + val
        # print(nutritions)
        breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.iloc[nutritions]
        breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
        # print("brfst=",breakfastfoodseparatedIDdata)



        # converting into numpy array
        breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
        # print("bstnumpi=",breakfastfoodseparatedIDdata)


        #print("len=",len(breakfastfoodseparatedIDdata))

        ## K-Means Based  Breakfast Food
        Datacalorie = breakfastfoodseparatedIDdata[0:, 1:len(breakfastfoodseparatedIDdata)]
        # print("DC=", Datacalorie)
        # print("DClwen=", len(Datacalorie))

        X = np.array(Datacalorie)
        # print("x=", X)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        y_kmeans = kmeans.predict(X)
        # print("y_kmeans=",y_kmeans)
        '''for i in range(len(y_kmeans)):
            print("xxx=", X[y_kmeans[i]])'''

        # XValu = np.arange(0, len(kmeans.labels_))
        # print("val=", XValu)

        # retrieving the labels for breakfast food
        brklbl = kmeans.labels_
        # print("brklbl=", brklbl)

        inp = []
        ## Reading of the Dataet
        datafin_nutr = pd.read_csv('nutrition_distriution2.csv')
        ## train set
        datafin_nutr = datafin_nutr.T
        # print("datafin_nutr=",datafin_nutr)


        weightlosscat = datafin_nutr.iloc[[1, 2, 7, 8]]

        weightlosscat = weightlosscat.T

        # Converting numpy array
        weightlosscatDdata = weightlosscat.to_numpy()

        weightlosscat = weightlosscatDdata[0:, 0:len(weightlosscatDdata)]


        weightlossfin = np.zeros((len(weightlosscat) * 5, 4), dtype=np.float32)
        # print("weightlossfin=",weightlossfin)
       # print("weightlossfinlen=", len(weightlossfin))

        t = 0
        r = 0
        s = 0
        yt = []
        yr = []
        ys = []

        #print("weightlosscat=", len(weightlosscat))

        for zz in range(5):
            # print("zz=", zz)
            for jj in range(len(weightlosscat)):
                # print("jj=",jj)
                valloc = list(weightlosscat[jj])
                # print("nval=", np.array(valloc))
                weightlossfin[t] = np.array(valloc)
                # print("brklbl=", brklbl[jj])
                yt.append(brklbl[jj])
                t += 1


        X_train = weightlossfin  # Features
        y_train = yt  # Labels

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=52)

        clf = RandomForestClassifier(n_estimators=100,max_depth=10)

        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy_bf = accuracy_score(y_test, y_pred) * 100
        print("BreakFast_accuracy=",accuracy_bf)
        return accuracy_bf




    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)


#Breakfast_accuracy()
