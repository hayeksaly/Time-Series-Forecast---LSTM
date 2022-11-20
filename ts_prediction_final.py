
# Time-series Prediction
---
## Case Study: Energy Consumption Prediction in Households
---
## Table of Contents 

### Section 1. Exploratory Data Analysis
'''
* 1.1: Data Loading and Understanding 
* 1.2: Data Cleaning 
* 1.3: Data Visualization

### Section 2. Time-series Data Preparation for Forecasting

### Section 3. Building Time-series Models

### Section 4. Evaluating Time-series Models
---

## 0. Scenario, Problem & Dataset Description

Dataset collected from a home in Paris, of the household owner's energy consumption patterns. The dataset collects around 50,000 measurements between December 2006 and November 2008 (so we have months of data to work with).

### The data is collected from multiple smart meters and contains the following attributes/features:

- date: Date in format dd/mm/yyyy

- time: time in format hh:mm:ss

- global_active_power: household global minute-averaged active power (in kilowatt)

- global_reactive_power: household global minute-averaged reactive power (in kilowatt)

- voltage: minute-averaged voltage (in volt)

- global_intensity: household global minute-averaged current intensity (in ampere)

- sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).

- sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.

- sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.

### Some notes about the dataset are:
1.  (global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3.

2.   The dataset contains some missing values in the measurements (nearly 1.25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. For instance, the dataset shows *missing values on April 28, 2007*.

Dataset Description [here](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption).

# **1. Exploratory Data Analysis**

### Importing Libraries 

First off, let's import the python libraries we know we'll typically be using.

Some of the common ones are:

*   [numpy](https://www.numpy.org/) - supports large, multidimensional arrays and has a lot of useful mathematical built-in functions to run on these arrays
*   [pandas](https://pandas.pydata.org/) - offers high-performance, easy-to-use data structures (e.g. can store data of multiple types in one data structure)
*   [matplotlib](https://matplotlib.org/) - 2D plotting library
"""
'''
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

""" Data Loading """

data = pd.read_csv('/content/household_power_consumption.txt', sep=';') 

# quickly visualize - print first 5 rows
data.head()

"""#### Some points to pay attention to:

*   Our data is collected every minute. 
*   We have measurements collected in different units. 
*   What are our descriptors? What is our target?

To answer the last question above, we need to define our problem. What are we trying to do?

We can choose to frame the problem as one of the following:

1.  We can predict a household's total power consumption. This can be useful for:

> *   Demand planning by power distribution companies
> *   Powering/connecting to renewable energy sources 
> *   Financial planning of users

2.  We can predict the household's consumption of specific power devices. This can be useful for:

> *   Users can better plan their consumption of loads
> *   Smart grids can switch off low-demand devices during peak demand hours (e.g. fridge)

Today, we will be answering a specific question:

***Given recent power consumption, what is the expected total power consumption for the next week?***

### Data Inspection
"""

# inspect types
data.info()
#data.dtypes

# let's get a look at summary of statistics
data.describe()

data.describe(include='all')

# data shape
rows, cols = data.shape
print(f"The dataset is composed of {rows} rows and {cols} columns.")

"""---

### Data Cleaning

In the `describe()` summary, we could see a symbol "?" showing in our data. Perhaps this is what the data collectors used to resemble missing values. Let's check April 29 of 2007 that we already know has a missing value.

We can make use of the date and time columns to easily search through our data. We will turn the date and time to indicies to easily navigate!
"""

data = pd.read_csv("household_power_consumption.txt", sep=';', infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime']) 
data.head()

"""Now we can easily search our data. """

import datetime 
# search for April 29, 2007 using the datetime index on the dataframe
data.loc[datetime.datetime(year=2007,month=4,day=29)]

"""**Finding:** "?" is a symbol used to represent missing values! 

Let's take care of these missing values.
"""

# Removing rows with missing values or "?"

# First. let's convert "?"" to nan
data.replace('?', np.nan, inplace=True)

# Let's check April 29, 2007 to see if we have properly replaced "?" with NaN
data.loc[datetime.datetime(year=2007,month=4,day=29)]

# So our data should be all float types, let's check
data.dtypes

# We need to change the data from string to float
data = data.astype('float32')
data.dtypes

# Second. let's check how many missing values we have
data.isnull().sum()

# One solution we could use is to drop all rows with missing values
data_remove = data.dropna()   
data_remove.shape

# Let's check April 29, 2007 to see if we have properly removed missing values from the dataset
#data_remove.loc[datetime.date(year=2007,month=4,day=29)]

data_remove.isnull().any()

# let's replace the missing values with the mean
data = data.fillna(data.mean())

# let's check if we have correctly replaced April 29, 2007 with the mean
data.loc[datetime.datetime(year=2007,month=4,day=29)]

# Any missing values left?
data.isnull().any()

"""**DONE** We have cleaned our data from missing values. 

---

### Data Preprocessing

Each measurement/sample in a row is collected every minute of the day. Do we need all this information?

If we're doing a prediction for the next week level consumptions, then we don't need to know how much power is being consumed per minute (too specific). We could probably get by with the consumption level per hour. We can likely even get by per day.
"""

# we can easily resample minutes to days
daily_data = data.resample('D', level=0).sum() #lean lday huwe at 0 level bel data set

# check new size of data (remember we lumped readings so data will shrink by an order of 1/(60 minutes per hour *24 hours per day)) 
daily_data.shape

# let's take a look at how our index looks like now
daily_data.head()

"""We now have our data sampled per day and in the proper setup for **one week ahead prediction.**

### Data Visualization
"""

daily_data_meter_1 = daily_data['Sub_metering_1']
daily_data_meter_2 = daily_data['Sub_metering_2']
daily_data_meter_3 = daily_data['Sub_metering_3']

daily_data_meter_1.plot(color="blue") # can plot directly in pandas

"""We can see how power consumption looks like for kitchen appliances."""

# Can plot them all in one figure directly from pandas
daily_data_per_meter = daily_data[['Sub_metering_1','Sub_metering_2','Sub_metering_3']]
daily_data_per_meter.plot()

"""It seems like `sub_metering_3` (which refers to heating and air conditioning) seems to have a unique pattern compared to the remaining measurements."""

# Using matplotlib
plt.figure()                # the first figure
plt.figure(figsize=(10,5))  # define figure size (rows, cols)
plt.suptitle('Usage of Kitchen, Laundry Room, and Heating Units Appliances')

plt.subplot(311)             # the first subplot in the first figure
plt.plot(daily_data['Sub_metering_1'],color="blue")
plt.xlabel('Time')
plt.ylabel('Watt-hour')

plt.subplot(312)             # the second subplot in the first figure
plt.plot(daily_data['Sub_metering_2'],color="orange")
plt.xlabel('Time')
plt.ylabel('Watt-hour')

plt.subplot(313)             # the third subplot in the first figure
plt.plot(daily_data['Sub_metering_3'],color="green")
plt.xlabel('Time')
plt.ylabel('Watt-hour')

"""**Question:** Do you notice anything about `sub_metering_3`?

**Finding:** There is a clear seasonal component. This is due to high consumption of heating appliances during winter and low consumption during summers. This behavior would look different if we were collecting measurements from UAE, for example.

# **2. Time-series Data Preparation**

OK, so now we should split our data into train and test sets. How do we do that? **Which columns are our descriptors and where's our target?**
"""

# The first and last days in the data are:
daily_data.iloc[[0]] # first day

daily_data.iloc[[-1]] # last day

"""Starting date: 2006-12-16
Ending date: 2008-01-28	

If we want one week ahead prediction, it makes sense to use the previous week of data to predict the week ahead. 

To set this up properly, we would want our week to start on Monday and end on Sunday.

> The **first Monday in the dataset** is December 18, 2006 (which is the third row in the dataset).

> The **last Sunday in the dataset** is January 28, 2008. (which is the -6 from the end).

Organizing the data into the step up above, we would have a total of 48 weeks.

*NOPE.. didn't count them myself! Here's an online [calculator](https://planetcalc.com/274/?license=1).*
"""

daily_data.iloc[[2]] # First day for us

daily_data.iloc[[-5]] # Last day for us

"""
> **Training & Testing samples are NOT randomly selected. Wait, what?!**

Unlike other datasets, we will not randomly select our test and train samples. We will divide our dataset into the most recent readings being training samples and the later readings testing samples. 

This is because we are working on a prediction problem, and thus, the sequence in which data is presented matters!"""

data = daily_data.values
train, test = data[2:-432], data[-432:-5] # Important numbers (ignoring the lingering days that don't fit into our Monday-Sunday structure)

# reshape into windows of weekly data (one week = 7 days) (total days / 7 = total weeks)
train = np.array(np.split(train, len(train)/7))
test = np.array(np.split(test, len(test)/7))

# check shape
print(f"The training set is {train.shape} and the test set is {test.shape}.")

"""144 weeks divided 7 days then 7 features

Where do these 3 dimensions come from? 
**(weeks, days, features)**

---

Here, we would usually normalize our data columns since we don't want our model to get biased towards specific features.

However, this is not NECESSARY when dealing with time-series prediction as our *features* are the data observations themselves.

---

So, now we have the training and testing sets. One more thing we need to prepare before moving to modeling is:

**How do we setup the data for supervised learning?** That is, what is my *X* and what is my *y*?

At each instant, we want to feed the model a week, and predict the week ahead. We don't do this prediction for Mondays only; we do them for every day of the week.

Meaning, we want our input (X) and output (y) to look like:

```
[Input], [Output]
[d01, d02, d03, d04, d05, d06, d07], [d08, d09, d10, d11, d12, d13, d14]
[d02, d03, d04, d05, d06, d07, d08], [d09, d10, d11, d12, d13, d14, d15]
Etc ...
```
"""

# flatten the train data over all weeks 
train = train.reshape((train.shape[0]*train.shape[1], train.shape[2])) # shape: [weeks*days, sensors] #nbr of days * number of weeks baaden lfeetures
# check shape
train.shape

# flatten the test data over all weeks 
test = test.reshape((test.shape[0]*test.shape[1], test.shape[2])) # shape: [weeks*days, sensors]
# check shape
test.shape

def supervised_setup(data, column):
  # data: expects train/test set with 2 dimensions of (samples, features)- i.e. one sensor reading
  # column: expects integer indicating column number of meter of interest

  X, y = [], [] # start with empty lists for X and y
  input_start = 0 # iterator
  n_input = 7 # we want 7 days as input
  n_out = 7 # we want 7 days as output

  # step over the entire history one time step at a time
  for i in range(len(data)):
	  # define the end of the input and corresponding output
	  input_end = input_start + n_input
	  output_end = input_end + n_out
	
    # ensure we have enough data for this instance
	  if output_end < len(data):
		  x_input = data[input_start:input_end, column]
		  x_input = x_input.reshape((len(x_input), 1))
  
		  X.append(x_input)
		  y.append(data[input_end:output_end, column])
  
	  # move along one time step
	  input_start += 1
  return np.array(X), np.array(y)

# TRY IT:

# split the data for the "total active power", i.e. column = 0
X_train, y_train = supervised_setup(train, 0)
X_test, y_test = supervised_setup(test, 0)

"""# **3. Building Time-series Models**

Recall, our dataset has different measures. Let's look at them again.
"""

daily_data.head()

"""**What is it that we want to predict?**

*   Univariate Forecasting - choose one specific measure that you want to carry out predictions on. Only need one model. 
*   Multivariate Forecasting - carry out predictions on multiple measures. Build a model for each measure OR feed each as a separate channel to the neural network (remember how we treat RGB channels in computer vision).

**First.** We will build a univariate prediction model to predict the next week consumption levels of total active power in the household (that is, column 0). Notice, predicting the next week consumption, day by day, is a multi-step ahead prediction where we are predicting SEVEN future values at once. 

If we were doing one day ahead prediction, this would correspond to one-step ahead prediction.


"""

# split the data for "total active power" -- (we had already done this previously)
X_train, y_train = supervised_setup(train, 0)  #0 hiye ltarget column taba3na
X_test, y_test = supervised_setup(test, 0)

"""Let's build our prediction model!

**What type of neural network would we use here?**
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

"""We will run a Long-Short Term Memory (LSTM) network since we are dealing with sequential data *(similar to dealing with language)*."""

# let's define some parameters for our LSTM model
n_inputs, n_channels, n_outputs = X_train.shape[1], X_train.shape[2], X_train.shape[1] #kam badna na3mel forecasting whon univariate la wehde so n_channnels is 1
n_cells, n_neurons = 64, 20

# build the network archiecture
model = Sequential()
model.add(LSTM(n_cells, activation='relu', input_shape=(n_inputs, n_channels))) # set return_sequences=True to add new LSTM layer  
#model.add(LSTM(n_cells, activation='relu', return_sequences=True))
#model.add(LSTM(n_cells))
model.add(Dense(n_neurons, activation='relu'))
model.add(Dense(n_outputs))

# compile the model
# we define the mean squared error as an evaluation metric for training & define ADAM as an optimization algorithm 
model.compile(loss='mse', optimizer='adam')
	
# train the network
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

"""# **4. Evaluating Time-series Models**

Recall, we want our input (X) and output (y) to look like:

```
[Input], [Output]
[d01, d02, d03, d04, d05, d06, d07], [d08, d09, d10, d11, d12, d13, d14]
[d02, d03, d04, d05, d06, d07, d08], [d09, d10, d11, d12, d13, d14, d15]
Etc ...
```

#### Walk-forward validation setup
"""

# make a single prediction
def forecast(model, history, n_input):
	# flatten data
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	
  # retrieve last observations for input data
	input_x = data[-n_input:, 0]
	
  # reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	
  # forecast the next week
	predicted_y = model.predict(input_x, verbose=0)
	
  # we only want the vector forecast
	predicted_y = predicted_y[0]
	return predicted_y

# To start evaluating our model, we can start from the last week in our training data
history = [x for x in X_train] # converting history into a list
n_input = 7

# We do walk-forward validation for each week
predictions = []
	
for i in range(len(X_test)):
	# predict the week
	y_predicted = forecast(model, history, n_input) 
	# collect predictions
	predictions.append(y_predicted)
	# get real observation and add it to my history 
	history.append(X_test[i,:])

"""#### Model Evaluation"""

from sklearn.metrics import mean_squared_error

# Now, we want to evaluate our model
predictions = np.array(predictions) # converting from list to np.array

# But we want to see how well our model is doing day by day 
scores = []
# calculate an RMSE score for each day
for i in range(y_test.shape[1]): # Loop over the days of each week (shape[1] refers to the days)
  # calculate mse for each day
  mse = mean_squared_error(y_test[:, i], predictions[:, i])
  rmse = np.sqrt(mse)
	# store
  scores.append(rmse)
 
# calculate overall RMSE for the entire week
weekly_score = np.array(scores).mean()

# print and plot scores
days = ['mon', 'tue', 'wed', 'thr', 'fri', 'sat', 'sun']
print('[AVG] %s: \n [%.3f] %s' % (days, weekly_score, scores))

plt.plot(days, scores, marker='o', label='lstm')
plt.show()

"""### Visual validation of model performance: Predictions versus Observations"""

# storing observations and model outputs in dataframes
true = pd.DataFrame(y_test[:,0]) # first day 2009-09-21
pred = pd.DataFrame(predictions[:,0])

# creating a datetime index for the dataframes
dti = pd.date_range('2009-09-21', periods=427, freq='D') # first day of test samples: 2009-09-21

# overlay observed sequence and predicted sequence
ax = true.plot()
pred.plot(ax=ax, alpha=.8, figsize=(14, 7))

xi = list(range(len(dti)))
plt.xticks(np.arange(min(xi), max(xi), 50), dti.date[np.arange(min(xi), max(xi), 50)])

ax.set_xlabel('Date')
ax.set_ylabel('Global Active Power (kW/day)')
plt.legend(['observed','predictions'])
plt.show()

"""**References**

The dataset was downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption#).

Tutorial inspired by [Machine Learning Mastery](https://machinelearningmastery.com/multi-step-time-series-forecasting-with-machine-learning-models-for-household-electricity-consumption/)
"""