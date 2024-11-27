
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import sys
from DBConnection import DBConnection
def Diet_Control_Breakfast(age,gender,height,weight,cuisine,catgry,sugar,bp,wl,cols,sno,allergitic_items):
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

        # calculating BMI
        age = int(age)
        gen = int(gender)
        diet = catgry
        cuisine = cuisine
        sugar = sugar  # 1-yes, 0-no
        bp = bp  # 1-yes, 0-no
        weight = int(weight)
        height = int(height)

        bmi = weight / ((height / 100) ** 2)
        # bmi2 = weight / (height** 2)
        # print(bmi2)
        # print(bmi)
        agewiseinp = 0
        agecl=0

        # Age Classs [1,2,3]
        for lp in range(0, 80, 20):
            # print("lp=",lp)
            test_list = np.arange(lp, lp + 20)
            # print("test_list=", test_list)
            for i in test_list:
                if (i == age):
                    tr = round(lp / 20)
                    agecl = round(lp / 20)
                    # print("tr=",tr)
                    # print("agecl=",agecl)

        print("Your body mass index is: ", bmi)
        # BMI class
        if (bmi < 16):
            # print("Acoording to your BMI, you are Severely Underweight")
            clbmi = 4
        elif (bmi >= 16 and bmi < 18.5):
            # print("Acoording to your BMI, you are Underweight")
            clbmi = 3
        elif (bmi >= 18.5 and bmi < 25):
            # print("Acoording to your BMI, you are Healthy")
            clbmi = 2
        elif (bmi >= 25 and bmi < 30):
            # print("Acoording to your BMI, you are Overweight")
            clbmi = 1
        elif (bmi >= 30):
            # print("Acoording to your BMI, you are Severely Overweight")
            clbmi = 0

        # converting into numpy array
        breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
        # print("bstnumpi=",breakfastfoodseparatedIDdata)

        ti = (clbmi + agecl + gen) / 2
        # print("ti=", ti)

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

        bmicls = [0, 1, 2, 3, 4]
        agecls = [0, 1, 2, 3, 4]

        weightlosscat = datafin_nutr.iloc[wl]
        # print("wlcat=", weightlosscat)
        weightlosscat = weightlosscat.T

        # Converting numpy array
        weightlosscatDdata = weightlosscat.to_numpy()

        weightlosscat = weightlosscatDdata[0:, 0:len(weightlosscatDdata)]

        # print("wlcatdata=", weightlosscat)

        weightlossfin = np.zeros((len(weightlosscat) * 5, cols), dtype=np.float32)
        # print("weightlossfin=",weightlossfin)
        print("weightlossfinlen=", len(weightlossfin))

        t = 0
        r = 0
        s = 0
        yt = []
        yr = []
        ys = []

        print("weightlosscat=", len(weightlosscat))

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
        print("yt=", len(yt))
        X_test = np.zeros((len(weightlosscat), cols), dtype=np.float32)
        # print("X_test=",X_test)

        # print('####################')

        # randomforest
        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            # valloc.append(agecl)
            # valloc.append(clbmi)

            X_test[jj] = np.array(valloc) * ti

        X_train = weightlossfin  # Features
        y_train = yt  # Labels

        # print("x_train=",X_train)

        # print("y_train=", y_train)
        # print("y_trainlen=", len(y_train))

        # print("X_test=", X_test)

        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=100)

        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)

        # print (X_test[1])
        # X_test2 = X_test
        y_pred = clf.predict(X_test)
        # print("y_pred=",y_pred)
        # print("y_predlen=", len(y_pred))
        # print("ids=",breakfastfoodseparatedID)
        # print("idslen=", len(breakfastfoodseparatedID))

        rec = []
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                rec.append(breakfastfoodseparatedID[i])

        # print("rec=",rec)

        # print(food_dataset.iloc[rec])

        food_dataset = food_dataset.iloc[rec]

        # print(food_dataset["Food_items"])

        diet_list = food_dataset["Diet"].tolist()
        sugar_list = food_dataset["Sugar"].tolist()
        bp_list = food_dataset["BP"].tolist()
        cu_list = food_dataset["Cuisine"].tolist()

        # print("diet_list=",len(diet_list))
        # print("rec_len=",len(rec))
        diet_rec = []

        for i in range(len(diet_list)):
            if diet_list[i] == diet and sugar_list[i] == sugar and bp_list[i] == bp and cu_list[i] == cuisine:
                diet_rec.append(rec[i])

        # print(breakfastfoodseparatedID)

        # print(diet_rec)
        # print(rec)
        # print(food_dataset)
        food_rec_brkfst = rec_food.iloc[diet_rec]
        recp_items = food_rec_brkfst["Food_items"].tolist()
        recp_inst = food_rec_brkfst["TranslatedInstructions"].tolist()

        database = DBConnection.getConnection()
        cursor = database.cursor()

        alrgitm=allergitic_items.split(",")


        print("len=",len(alrgitm))
        print("BreakFast Recipe List:")
        print("===========================")
        print("Recipe Name" + "\t" + "INSTRUCTIONS")

        for k in range(len(recp_items)):
                cnt = 0
            #brkfast.clear()
                print(recp_items[k] + "\t" + recp_inst[k])

                for i in alrgitm:
                    #print("i=",i)
                    recipedesc=recp_inst[k]
                    #print("recipedesc=",recipedesc)

                    if i.lower() in recipedesc.lower():
                        cnt=cnt+1

                print("cnt=",cnt)
                if cnt == 0:
                    sql = "insert into food_recommends values(%s,%s,%s,%s)"
                    values = (str(sno), str(recp_items[k]), str(recp_inst[k]), "Breakfast")
                    cursor.execute(sql, values)
                    database.commit()

                    sql = "insert into temp values(%s,%s,%s)"
                    values = (str(recp_items[k]), str(recp_inst[k]), "Breakfast")
                    cursor.execute(sql, values)
                    database.commit()




    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)






'''wl=[1, 2, 7, 8]
cols=4
Diet_Control_Breakfast(32,1,120,78,"South Indian Recipes","Vegetarian",1,1,wl,cols,1)'''














