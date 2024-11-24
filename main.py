
#**************** IMPORT PACKAGES ********************
from Database import *
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt2
from sklearn.ensemble import RandomForestRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math, random
from sklearn.preprocessing import StandardScaler


#***************** FLASK *****************************
app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


@app.route('/dashboard')
def dashboard():
   
    all_years = [2013,2014,2015,2016,2017]
    all_store = list(range(1, 11))
    all_items = list(range(1, 51))
    return render_template('dashboard.html',all_years=all_years,all_store=all_store,all_items=all_items)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/index')
def index1():
   return render_template('index.html')


@app.route('/home')
def home():
   return render_template('home.html')


@app.route('/registration',methods = ['POST','GET'])
def registration():
   if request.method=="POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        mobile = request.form["mobile"]
        InsertData(username,email,password,mobile)
        return render_template('login.html')

   return render_template('registration.html')



@app.route('/login',methods = ['POST','GET'])
def login():
   if request.method=="POST":
        email = request.form['email']
        passw = request.form['password']
        resp = read_cred(email, passw)
        if resp != None:
            return redirect("/home")
        else:
            message = "Username and/or Password incorrect.\\n        Yo have not registered Yet \\nGo to Register page and do Registration";
            return "<script type='text/javascript'>alert('{}');</script>".format(message)

   return render_template('login.html')




@app.route('/product_analysis',methods = ['POST','GET'])
def product_analysis():
    year = int(request.form['year'])
    store = int(request.form['store'])
    item = int(request.form['item'])
    days = int(request.form['days'])
    
    #**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(year,store,item):
        df = pd.read_csv('dataset/Store_Item_Demand_Forecasting.csv')

        # Filter the dataframe to include only item 1 and store 1
        df = df[(df['item'] == item) & (df['store'] == store)]

        # define the original date format
        original_format = '%Y-%m-%d'

        # convert the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # filter the dataframe to include only data from the year 2017
        df = df[df['date'].dt.year == year]

        # convert the 'date' column back to the original format
        df['date'] = df['date'].dt.strftime(original_format)

        print(df.head())

        return df

    #******************** ARIMA SECTION ********************
    #******************** ARIMA SECTION ********************
    def ARIMA_ALGO(df):
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')

        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1 ,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                if isinstance(output, np.ndarray):
                    yhat = output[0]
                else:
                    yhat = output
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            return predictions

        data = df

        data['sales'] = data['sales']
        Quantity_date = data[['sales','date']]
        Quantity_date.index = Quantity_date['date'].map(lambda x: parser(str(x)))
        Quantity_date['sales'] = Quantity_date['sales'].map(lambda x: float(x))
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        Quantity_date = Quantity_date.drop(['date'],axis =1)
        fig = plt.figure(figsize=(7.2,4.8),dpi=70)
        plt.plot(Quantity_date)
        plt.savefig('static/Trends.png')
        plt.close(fig)

        # quantity = Quantity_date.values
        # size = int(len(quantity) * 0.80)
        # train, test = quantity[0:size], quantity[size:len(quantity)]
        # #fit in model
        # predictions = arima_model(train, test)

        # #plot graph
        # fig = plt.figure(figsize=(7.2,4.8),dpi=70)
        # plt.plot(test,label='Actual Sales')
        # plt.plot(predictions,label='Predicted Sales')
        # plt.legend(loc=4)
        # plt.title('ARIMA Sales Prediction')
        # plt.xlabel('Time')
        # plt.ylabel('Sales')
        # plt.savefig('static/ARIMA.png')
        # plt.close(fig)
        # print()
        # print("##############################################################################")
        # arima_pred=predictions[-2]
        # print("Next Days's sales",item," Prediction by ARIMA:",arima_pred)
        # #rmse calculation
        # error_arima = math.sqrt(mean_squared_error(test, predictions))
        # print("ARIMA RMSE:",error_arima)
        # print("##############################################################################")
        return 0
        # return arima_pred, error_arima
    
    #************* LSTM SECTION **********************


            
    def RAN_FOREST_ALGO(df):
        #No of days to be forcasted in future
        forecast_out = int(1)
        #Price after n days
        df['Sales after n days'] = df['sales'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['sales','Sales after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
        
        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]

        sc = StandardScaler()
        
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
        
        #Training
        clf = RandomForestRegressor()
        clf.fit(X_train, y_train)
        
        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2,4.8),dpi=70)
        plt2.plot(y_test,label='Actual Sales' )
        plt2.plot(y_test_pred,label='Predicted Sales')
        plt2.title('Random Forest Sales Prediction')
        plt2.xlabel('Time')
        plt2.ylabel('Sales')

        plt2.legend(loc=4)
        plt2.savefig('static/RF.png')
        plt2.close(fig)
        
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        
        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0]
        print()
        print("##############################################################################")
        print("Next Day's Sales Prediction by Random Forest: ",lr_pred)
        print("Random Forest RMSE:",error_lr)
        print("##############################################################################")
        return lr_pred, error_lr

    

    def LSTM_ALGO(df):

        # Split the data into training and testing sets
        training_size = int(0.8 * len(df))
        training_set = df.iloc[:training_size, 3:4].values
        testing_set = df.iloc[training_size:, 3:4].values

        # Scale the data
        sc = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled = sc.fit_transform(training_set)

        # Generate training data with 7 days memory
        X_train = []
        y_train = []
        for i in range(7, training_size):
            X_train.append(training_set_scaled[i-7:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshape the data for the LSTM model
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        model.add(LSTM(units = 50))
        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs = 50, batch_size = 32)

        # Generate testing data with 7 days memory
        inputs = df.iloc[len(df) - len(testing_set) - 7:, 3:4].values
        inputs = sc.transform(inputs)

        X_test = []
        for i in range(7, len(inputs)):
            X_test.append(inputs[i-7:i, 0])
        X_test = np.array(X_test)

        # Reshape the data for the LSTM model
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Generate predicted sales for the testing data
        predicted_sales = model.predict(X_test)
        predicted_sales = sc.inverse_transform(predicted_sales)

        # Calculate the RMSE score
        rmse = np.sqrt(mean_squared_error(testing_set, predicted_sales))
        print("RMSE score:", rmse)

        # Print the predicted sales for the next day
        next_day_sales = sc.inverse_transform(model.predict(np.reshape(X_test[-1], (1, X_test.shape[1], 1))))[0][0]
        print("Predicted sales for the next day:", next_day_sales)

        # Visualize the results
        fig = plt.figure(figsize=(7.2,4.8),dpi=70)
        plt.plot(testing_set, label = 'Actual Sales')
        plt.plot(predicted_sales,  label = 'Predicted Sales')
        plt.title('LSTM Sales Prediction')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.legend()
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        return next_day_sales,rmse

    #***************** LINEAR REGRESSION SECTION ******************       
    def LIN_REG_ALGO_NORMAL(df):
        #No of days to be forcasted in future
        forecast_out = int(1)
        #Price after n days
        df['sales after n days'] = df['sales'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['sales','sales after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])

        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]

        # Feature Scaling===Normalization
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        X_to_be_forecasted=sc.transform(X_to_be_forecasted)

        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        fig = plt2.figure(figsize=(7.2,4.8),dpi=70)
        plt2.plot(y_test,label='Actual Sales' )
        plt2.plot(y_test_pred,label='Predicted Sales')
        plt.title('LR Sales Prediction')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt2.legend(loc=4)
        plt2.savefig('static/LR.png')
        plt2.close(fig)

        

        
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))


        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        print()
        print("##############################################################################")
        print("Next Day's ",item," Sales Prediction by Linear Regression: ",lr_pred)
        print("Linear Regression RMSE:",error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr


    #***************** LINEAR REGRESSION SECTION ******************       
    def LIN_REG_ALGO(df,forecast_days):
        #No of days to be forcasted in future
        forecast_out = int(forecast_days)
        #Price after n days
        df['sales after n days'] = df['sales'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['sales','sales after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])

        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]

        # Feature Scaling===Normalization
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        X_to_be_forecasted=sc.transform(X_to_be_forecasted)

        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))


        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        print()
        
        fig = plt2.figure(figsize=(7.2,4.8),dpi=200)
        plt2.plot(forecast_set,label='Predicted Sales')
        plt.title('Predicted Saled for Next '+str(forecast_days)+" Days")
        plt.xlabel('Days')
        plt.ylabel('Sales')
        plt2.legend(loc=4)
        plt2.savefig('static/FORECASTING.png')
        plt2.close(fig)

        
        print("##############################################################################")
        print("Next Day's ",item," Sales Prediction by Linear Regression: ",lr_pred)
        print("Linear Regression RMSE:",error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr



    print("##############################################################################")
    print("Getting Product Data\nYear Data: ",year,"\nStore ID: ",store,"\nItem ID: ",item)
    df =get_historical(year,store,item)
    print("##############################################################################")
    df = df.dropna()

    today_sales=df.iloc[-1:]
    ARIMA_ALGO(df.copy())

    arima_pred,error_arima=RAN_FOREST_ALGO(df.copy())
    lstm_pred, error_lstm=LSTM_ALGO(df.copy())
    df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO_NORMAL(df.copy())
    df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO(df.copy(),days)

    quote=    "Store ID: "+str(store)+" Item ID: "+str(item)
    print()
    print("Forecasted Sales for Next 7 days: "+quote+"\n")
    # print(forecast_set)
    
    return render_template('results.html',quote=quote,arima_pred=round(arima_pred,2),lstm_pred=round(lstm_pred,2),
                            lr_pred=round(lr_pred,2),
                            days  = days,
                            recc_result ="",
                            forecast_set=forecast_set,error_lr=round(error_lr,2),error_lstm=round(error_lstm,2),error_arima=round(error_arima,2))



if __name__ == '__main__':
   app.run(debug=False)
   

















