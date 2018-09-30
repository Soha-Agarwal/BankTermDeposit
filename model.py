import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree
#from sklearn import cross_validation
def preprocessing():
	bank=pd.read_csv("/home/krutee/Data_mining/bank.csv")
	bank_df = pd.DataFrame(bank)
	bank.head()



		#education column have basic.4y basic.y and basic.9y combining all of them in basic

		#bank['education'] = np.where(bank['education'] == 'basic.4y','basic',bank['education'])
		#bank['education'] = np.where(bank['education'] == 'basic.6y','basic',bank['education'])
		#bank['education'] = np.where(bank['education'] == 'basic.9y','basic',bank['education'])

		#X=bank.iloc[:,:-1].values
		#y=bank.iloc[:,-1].values

	cols =['job', 'marital', 'education', 'default', 'housing', 'loan',
				 'contact', 'month', 'day_of_week', 'poutcome']
	data_1 = bank[cols]
	data_dummies = pd.get_dummies(data_1)
	result_df = pd.concat([data_dummies, bank], axis=1)


	 # print(result_df.columns.values)

		#changing yes/no to 1/0
	result_df['output'] = result_df['y'].apply(lambda x: 1 if x =='yes' else 0)
	result_df['pdays'] = result_df['pdays'].apply(lambda x: 0 if x ==999 else x) # to remove 999
	result_df.rename(columns={'education_basic.4y': 'education_basic_4y', 'education_basic.6y':'education_basic_6y', 
														 'education_basic.9y': 'education_basic_9y',
														 'education_high.school': 'education_high_school',
														 'education_professional.course': 'education_professional_course',
														 'education_university.degree': 'education_university_degree',
														 'log_emp.var.rate': 'log_emp_var_rate',
														 'cons.price.idx': 'cons_price_idx',
														 'cons.conf.idx': 'cons_conf_idx',
														 'log_nr.employed': 'log_nr_employed',
														 'job_self-employed': 'job_self_employed',
														 'job_blue-collar': 'job_blue_collar',
														 'nr.employed': 'nr_employed'}, inplace=True)

	result_df2 = result_df[['job_admin.', 'job_blue_collar', 'job_entrepreneur',
					 'job_housemaid', 'job_management', 'job_retired',
					 'job_self_employed', 'job_services', 'job_student',
					 'job_technician', 'job_unemployed', 'job_unknown',
					 'marital_divorced', 'marital_married', 'marital_single',
					 'marital_unknown', 'education_basic_4y', 'education_basic_6y',
					 'education_basic_9y', 'education_high_school',
					 'education_illiterate', 'education_professional_course',
					 'education_university_degree', 'education_unknown', 'default_no',
					 'default_unknown', 'default_yes', 'housing_no', 'housing_unknown',
					 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
					 'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug',
					 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may',
					 'month_nov', 'month_oct', 'month_sep', 'day_of_week_fri',
					 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
					 'day_of_week_wed', 'poutcome_failure', 'poutcome_nonexistent',
					 'poutcome_success', 'age',
					 'cons_price_idx', 'cons_conf_idx',
					 'campaign', 'pdays', 'previous',
					 'euribor3m', 'nr_employed','output']]
	y = result_df2['output'].values
	X = result_df2[['job_admin.', 'job_blue_collar', 'job_entrepreneur',
					 'job_housemaid', 'job_management', 'job_retired',
					 'job_self_employed', 'job_services', 'job_student',
					 'job_technician', 'job_unemployed', 'job_unknown',
					 'marital_divorced', 'marital_married', 'marital_single',
					 'marital_unknown', 'education_basic_4y', 'education_basic_6y',
					 'education_basic_9y', 'education_high_school',
					 'education_illiterate', 'education_professional_course',
					 'education_university_degree', 'education_unknown', 'default_no',
					 'default_unknown', 'default_yes', 'housing_no', 'housing_unknown',
					 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
					 'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug',
					 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may',
					 'month_nov', 'month_oct', 'month_sep', 'day_of_week_fri',
					 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
					 'day_of_week_wed', 'poutcome_failure', 'poutcome_nonexistent',
					 'poutcome_success', 'age', 
					 'cons_price_idx', 'cons_conf_idx',
					 'campaign', 'pdays', 'previous',
					 'euribor3m', 'nr_employed']].values


		#split training and testing
	from sklearn.model_selection import train_test_split
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
		#feature Scaling
	from sklearn.preprocessing import MinMaxScaler
	scl_obj = MinMaxScaler(feature_range =(0, 1))
	scl_obj.fit(X_train) # find scalings for each column that make this zero mean and unit std
	X_train_scaled = scl_obj.transform(X_train) # apply to training
	X_test_scaled = scl_obj.transform(X_test) # apply those means and std to the test set (without snooping at the test set values)
	return X_train_scaled,X_test_scaled,y_train,y_test	
		#Data Preprocessing Complete

#Classification
#1.Logistic Regression
def logisticreg(X_train,X_test,y_train,y_test):
	from sklearn.linear_model import LogisticRegression
	classifier=LogisticRegression(penalty='l2', C=0.05, class_weight="balanced")
	classifier.fit(X_train,y_train)
	y_pred=classifier.predict(X_test)

	from sklearn.metrics import confusion_matrix
	cm=confusion_matrix(y_test,y_pred)
	 # print(cm)

	from sklearn import metrics as mt
	acc = mt.accuracy_score(y_test,y_pred)
	conf = mt.confusion_matrix(y_test,y_pred)
	print ('accuracy:', acc)
	print ("Confusion matrix : ")
	print (conf)

	print ("Classification Report..")
	from sklearn.metrics import classification_report
	print(classification_report(y_test, y_pred))
	return classifier


#y_pred1=classifier.predict()
#return y_pred1
#import ipywidgets as widgets
#text=widgets.Text()
#display(text)
#X_train=[[]],y_train=[[]]
#X_test,y_test
#preprocessing()
#y_pred=logisticreg()

#X_train,X_test,y_train,y_test=preprocessing()
#logisticreg(X_train,X_test,y_train,y_test)
#2.KNN

def KNN(X_train,X_test,y_train,y_test):
	from sklearn.neighbors import KNeighborsClassifier
	knnclassifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
	knnclassifier.fit(X_train,y_train)
	y_pred=knnclassifier.predict(X_test)

	from sklearn.metrics import confusion_matrix
	cm=confusion_matrix(y_test,y_pred)
   # print(cm)

	from sklearn import metrics as mt
	acc = mt.accuracy_score(y_test,y_pred)
	conf = mt.confusion_matrix(y_test,y_pred)
	print ('accuracy:', acc) 
	print ("Confusion matrix : ")
	print (conf)
	print
	print ("Classification Report..")
	from sklearn.metrics import classification_report
	print(classification_report(y_test, y_pred))
	return knnclassifier



def RandomForest(X_train,X_test,y_train,y_test):
	print('Random Forest Classifier')
	model_rf = RandomForestClassifier(max_depth = 8, n_estimators = 120)
	model_rf.fit(X_train, y_train)
	y_pred_rf = model_rf.predict_proba(X_test)[:, 1]
	from sklearn.metrics import confusion_matrix
	cm=confusion_matrix(y_test,y_pred)
   # print(cm)

	from sklearn import metrics as mt
	acc = mt.accuracy_score(y_test,y_pred)
	conf = mt.confusion_matrix(y_test,y_pred)
	print ('accuracy:', acc) 
	print ("Confusion matrix : ")
	print (conf)
	print
	print ("Classification Report..")
	from sklearn.metrics import classification_repor
	print(classification_report(y_test, y_pred))
	return model_rf
