from flask import Flask, render_template, request,flash
import pandas as pd
from flask import Response
import csv
from flask import session
from DBConnection import DBConnection
from Breakfast2 import Diet_Control_Breakfast
from Lunch2 import Diet_Control_Lunch
from Dinner2 import Diet_Control_Dinner
import sys
app = Flask(__name__)
app.secret_key = "abc"




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signin')
def signin():
    return render_template('user_signin.html')


@app.route('/user_home')
def user_home():
    return render_template('user_home.html')


@app.route('/signup')
def signup():
    return render_template('user_signup.html')



@app.route('/nutritional_FRS')
def nutritional_FRS():
    return render_template('food_recommendation.html')


@app.route('/feedback')
def feedback():
    return render_template('feedback.html')



@app.route('/save_feedback',methods =["GET", "POST"])
def save_feedback():
    try:
        feedback = request.form.get('fb')
        uid=session['uid']

        database = DBConnection.getConnection()
        cursor = database.cursor()

        sql = "insert into feedback values(%s,%s)"
        values = (uid, feedback)
        cursor.execute(sql, values)
        database.commit()

    except Exception as e:
        print(e)
    return render_template('feedback.html',messages="done")



@app.route('/registering',methods =["GET", "POST"])
def registering():
    try:
        name = request.form.get('name')
        uid = request.form.get('uid')
        pwd = request.form.get('pwd')
        email = request.form.get('email')
        mno = request.form.get('mno')

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:

            return render_template("user_signup.html", messages="User Id already exists..!")

        else:
            sql = "insert into register values(%s,%s,%s,%s,%s)"
            values = (name, uid, pwd, email, mno)
            cursor.execute(sql, values)
            database.commit()

        return render_template("user_signin.html", messages="Registered Successfully..! Login Here.")
    except Exception as e:
        print(e)
    return render_template('user_signin.html')


@app.route('/food_recommends',methods =["GET", "POST"])
def food_recommends():
    try:
        age = request.form.get('age')
        gender = request.form.get('gender')
        height = request.form.get('height')
        weight = request.form.get('weight')
        cuisine = request.form.get('cuisine')
        catgry = request.form.get('catgry')
        sugar = request.form.get('sugar')
        bp = request.form.get('bp')
        diet_catg = request.form.get('diet_catg')

        allergitic_items=request.form.get('alrgitem')


        database = DBConnection.getConnection()
        cursor = database.cursor()

        sql = "select max(sno) from history "
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res==None:
            res=0

        #print("ressss=",res)

        sno=res+1

        uid=session['uid']

        database = DBConnection.getConnection()
        cursor = database.cursor()

        cursor.execute("delete from temp")
        database.commit()

        if gender=="1":
            gen="Male"
        else:
            gen="FeMale"

        if sugar == "1":
            sugr = "Yes"
        else:
            sugr = "No"

        if bp == "1":
            bpp = "Yes"
        else:
            bpp = "No"

        if diet_catg == "1":
            dc = "WeightLoss"
        elif diet_catg == "2":
            dc = "Weightgain"
        else:
            dc = "Health"




        sql = "insert into history values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        values = (sno,age,gen,height,weight,cuisine,catgry,sugr,bpp,dc,uid)
        cursor.execute(sql, values)
        database.commit()
        breakfast_result=""

        print("diet_catg=",diet_catg)



        if diet_catg=="1":
            wl = [1, 2, 7, 8]
            cols = 4
            Diet_Control_Breakfast(age, gender, height, weight, cuisine, catgry, int(sugar),int(bp), wl, cols,sno,allergitic_items)
            Diet_Control_Lunch(int(age), int(gender), int(height), int(weight),cuisine, catgry, int(sugar), int(bp), wl, cols,sno,allergitic_items)
            Diet_Control_Dinner(int(age), int(gender), int(height), int(weight), cuisine, catgry, int(sugar),int(bp), wl, cols,sno,allergitic_items)
        elif diet_catg=="2":
            wl = [0, 1, 2, 3, 4, 7, 9, 10]
            cols = 8
            Diet_Control_Breakfast(age, gender, height, weight, cuisine, catgry, int(sugar), int(bp), wl, cols, sno,
                                   allergitic_items)
            Diet_Control_Lunch(int(age), int(gender), int(height), int(weight), cuisine, catgry, int(sugar), int(bp),
                               wl, cols, sno, allergitic_items)
            Diet_Control_Dinner(int(age), int(gender), int(height), int(weight), cuisine, catgry, int(sugar), int(bp),
                                wl, cols, sno, allergitic_items)


        else:
            wl = [1, 2, 3, 4, 6, 7, 9]
            cols = 7

            Diet_Control_Breakfast(age, gender, height, weight, cuisine, catgry, int(sugar), int(bp), wl, cols, sno,
                                   allergitic_items)
            Diet_Control_Lunch(int(age), int(gender), int(height), int(weight), cuisine, catgry, int(sugar), int(bp),
                               wl, cols, sno, allergitic_items)
            Diet_Control_Dinner(int(age), int(gender), int(height), int(weight), cuisine, catgry, int(sugar), int(bp),
                                wl, cols, sno, allergitic_items)


    except Exception as e:
        print("Error2=", e)
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)

    database = DBConnection.getConnection()
    cursor = database.cursor()
    cursor.execute("SELECT *FROM temp")
    rows = cursor.fetchall()

    cursor1 = database.cursor()
    cursor1.execute("SELECT *FROM feedback")
    rows1 = cursor1.fetchall()

    return render_template('recommendation_results.html', results=rows, feedback=rows1)


@app.route("/logincheck",methods =["GET", "POST"])
def userlogin():
        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        print(uid,pwd)
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "' and passwrd='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid

            return render_template("user_home.html")
        else:

            return render_template("user_signin.html",msg="Invalid Credentials")



        return render_template("admin.html")




@app.route('/history')
def history():
    database = DBConnection.getConnection()
    cursor = database.cursor()

    uid=session['uid']
    cursor.execute("SELECT *FROM history where userid='"+uid+"' ")
    rows = cursor.fetchall()

    return render_template('history.html',rawdata=rows)


@app.route("/view/<sno>/")
def view(sno):
    database = DBConnection.getConnection()
    cursor = database.cursor()

    cursor.execute("SELECT *FROM food_recommends where sno='"+sno+"' ")
    rows = cursor.fetchall()

    return render_template('recommendation_results2.html',results=rows)

if __name__ == '__main__':
    app.run(host="localhost", port=2468, debug=True)
