from flask import Flask,render_template,request,flash,json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from model import preprocessing

app=Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
	if request.method=='POST':
		form=request.form		
		age=form['age']
		marital=form['marital']
		default=form['default']
		housing=form['housing']
		loan=form['loan']
		contact=form['contact']
		month=form['month']
		day_of_week=form['day_of_week']
		duration=form['duration']
		campaign=form['campaign']
		pdays=form['pdays']
		previous=form['previous']
		poutcome=form['poutcome']
		job=form['job']
		education=form['education']
		emp_var_rate=1.1
		cons_price_idx=93.2
		cons_conf_idx=-42.7
		euribor3m=4.968
		nr_employed=form['employed']
		#marital
		if(marital=='married'):
			marital_single=0
			marital_married=1
			marital_divorced=0
			marital_unknown=0
		elif(marital=='single'):
			marital_single=1
			marital_married=0
			marital_divorced=0
			marital_unknown=0
		elif(marital=='divorced'):
			marital_single=0
			marital_married=0
			marital_divorced=1
			marital_unknown=0
		elif(marital=='unknown'):
			marital_single=0
			marital_married=0
			marital_divorced=0
			marital_unknown=1

		#default
		if(default=='yes'):
			default_yes=1
			default_no=0
			default_unknown=0
		elif(default=='no'):
			default_yes=0
			default_no=1
			default_unknown=0
		else:
			default_yes=0
			default_no=0
			default_unknown=1

		#housing
		if(housing=='yes'):
			housing_yes=1
			housing_no=0
			housing_unknown=0
		elif(housing=='no'):
			housing_yes=0
			housing_no=1
			housing_unknown=0
		else:
			housing_yes=0
			housing_no=0
			housing_unknown=1

		#loan
		if(loan=='yes'):
			loan_yes=1
			loan_no=0
			loan_unknown=0
		elif(loan=='no'):
			loan_yes=0
			loan_no=1
			loan_unknown=0
		else:
			loan_yes=0
			loan_no=0
			loan_unknown=1

		#contact
		if(contact=='cellular'):
			contact_cellular=1
			contact_telephone=0
		else:
			contact_cellular=0
			contact_telephone=1

		#job
		if(job=='admin'):
			job_admin=1
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='blue_collar'):
			job_admin=0
			job_blue_collar=1
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='entreprenuer'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=1
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='housemaid'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=1
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='management'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=1
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='retired'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=1
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='self_employed'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=1
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='services'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=1
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='student'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=1
			job_technician=0
			job_unemployed=0 
			job_unknown=0
		elif(job=='technician'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=1
			job_unemployed=0 
			job_unknown=0
		elif(job=='unemployed'):
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=1 
			job_unknown=0
		else:
			job_admin=0
			job_blue_collar=0
			job_entrepreneur=0
			job_housemaid=0
			job_management=0
			job_retired=0
			job_self_employed=0
			job_services=0
			job_student=0
			job_technician=0
			job_unemployed=0 
			job_unknown=1

			#education
		if(education=='basic.4y'):
			education_basic_4y=1
			education_basic_6y=0
			education_basic_9y=0
			education_high_school=0
			education_illiterate=0
			education_professional_course=0
			education_university_degree=0 
			education_unknown=0
		elif(education=='basic.6y'):
			education_basic_4y=0
			education_basic_6y=1
			education_basic_9y=0
			education_high_school=0
			education_illiterate=0
			education_professional_course=0
			education_university_degree=0 
			education_unknown=0
		elif(education=='basic.9y'):
			education_basic_4y=0
			education_basic_6y=0
			education_basic_9y=1
			education_high_school=0
			education_illiterate=0
			education_professional_course=0
			education_university_degree=0 
			education_unknown=0
		elif(education=='high_school'):
			education_basic_4y=0
			education_basic_6y=0
			education_basic_9y=0
			education_high_school=1
			education_illiterate=0
			education_professional_course=0
			education_university_degree=0 
			education_unknown=0
		elif(education=='illiterate'):
			education_basic_4y=0
			education_basic_6y=0
			education_basic_9y=0
			education_high_school=0
			education_illiterate=1
			education_professional_course=0
			education_university_degree=0 
			education_unknown=0
		elif(education=='professional_course'):
			education_basic_4y=0
			education_basic_6y=0
			education_basic_9y=0
			education_high_school=0
			education_illiterate=0
			education_professional_course=1
			education_university_degree=0 
			education_unknown=0
		elif(education=='university_degree'):
			education_basic_4y=0
			education_basic_6y=0
			education_basic_9y=0
			education_high_school=0
			education_illiterate=0
			education_professional_course=0
			education_university_degree=1 
			education_unknown=0
		else:
			education_basic_4y=0
			education_basic_6y=0
			education_basic_9y=0
			education_high_school=0
			education_illiterate=0
			education_professional_course=0
			education_university_degree=0 
			education_unknown=1
			
			#month
		if(month=='mar'):
			month_apr=0
			month_aug=0
			month_dec=0
			month_jul=0 
			month_jun=0 
			month_mar=1
			month_may=0
			month_nov=0
			month_oct=0 
			month_sep=0
		elif(month=='apr'):
			month_apr=1
			month_aug=0
			month_dec=0
			month_jul=0 
			month_jun=0 
			month_mar=0
			month_may=0
			month_nov=0
			month_oct=0 
			month_sep=0
		elif(month=='may'):
			month_apr=0
			month_aug=0
			month_dec=0
			month_jul=0 
			month_jun=0 
			month_mar=0
			month_may=1
			month_nov=0
			month_oct=0 
			month_sep=0
		elif(month=='jun'):
			month_apr=0
			month_aug=0
			month_dec=0
			month_jul=0 
			month_jun=1 
			month_mar=0
			month_may=0
			month_nov=0
			month_oct=0 
			month_sep=0
		elif(month=='jul'):
			month_apr=0
			month_aug=0
			month_dec=0
			month_jul=1 
			month_jun=0 
			month_mar=0
			month_may=0
			month_nov=0
			month_oct=0 
			month_sep=0
		elif(month=='aug'):
			month_apr=0
			month_aug=1
			month_dec=0
			month_jul=0 
			month_jun=0 
			month_mar=0
			month_may=0
			month_nov=0
			month_oct=0 
			month_sep=0
		elif(month=='sep'):
			month_apr=0
			month_aug=0
			month_dec=0
			month_jul=0 
			month_jun=0 
			month_mar=0
			month_may=0
			month_nov=0
			month_oct=0 
			month_sep=1
		elif(month=='oct'):
			month_apr=0
			month_aug=0
			month_dec=0
			month_jul=0 
			month_jun=0 
			month_mar=0
			month_may=0
			month_nov=0
			month_oct=1 
			month_sep=0
		elif(month=='nov'):
			month_apr=0
			month_aug=0
			month_dec=0
			month_jul=0 
			month_jun=0 
			month_mar=0
			month_may=0
			month_nov=1
			month_oct=0 
			month_sep=0


			#day
		if(day_of_week== 'mon'):
			day_of_week_fri=0
			day_of_week_mon=1
			day_of_week_thu=0
			day_of_week_tue=0
			day_of_week_wed=0
		elif(day_of_week== 'tue'):
			day_of_week_fri=0
			day_of_week_mon=0
			day_of_week_thu=0
			day_of_week_tue=1
			day_of_week_wed=0
		elif(day_of_week== 'wed'):
			day_of_week_fri=0
			day_of_week_mon=0
			day_of_week_thu=0
			day_of_week_tue=0
			day_of_week_wed=1
		elif(day_of_week== 'thu'):
			day_of_week_fri=0
			day_of_week_mon=0
			day_of_week_thu=1
			day_of_week_tue=0
			day_of_week_wed=0
		elif(day_of_week== 'fri'):
			day_of_week_fri=1
			day_of_week_mon=0
			day_of_week_thu=0
			day_of_week_tue=0
			day_of_week_wed=0

			#poutcome

		if(poutcome=='failure'):
			poutcome_failure=1 
			poutcome_nonexistent=0
			poutcome_success=0
		elif(poutcome=='success'):
			poutcome_failure=0 
			poutcome_nonexistent=0
			poutcome_success=1
		elif(poutcome=='nonexistent'):
			poutcome_failure=0 
			poutcome_nonexistent=1
			poutcome_success=0


		datapoint=[[]]
		datapoint=[np.array([job_admin, job_blue_collar, job_entrepreneur,
		   job_housemaid, job_management, job_retired,
		   job_self_employed, job_services, job_student,
		   job_technician, job_unemployed, job_unknown,
		   marital_divorced, marital_married, marital_single,
		   marital_unknown, education_basic_4y,education_basic_6y,
		   education_basic_9y, education_high_school,
		   education_illiterate, education_professional_course,
		   education_university_degree, education_unknown, default_no,
		   default_unknown, default_yes, housing_no, housing_unknown,
		   housing_yes, loan_no, loan_unknown, loan_yes,
		   contact_cellular, contact_telephone, month_apr, month_aug,
		   month_dec, month_jul, month_jun, month_mar, month_may,
		   month_nov, month_oct, month_sep, day_of_week_fri,
		   day_of_week_mon, day_of_week_thu, day_of_week_tue,
		   day_of_week_wed, poutcome_failure, poutcome_nonexistent,
		   poutcome_success, age, 
		   cons_price_idx, cons_conf_idx,
		   campaign, pdays, previous,
		   euribor3m, nr_employed])]
		
		scl_obj = MinMaxScaler(feature_range =(0, 1))
		scl_obj.fit(datapoint) # find scalings for each column that make this zero mean and unit std
		X_train_scaled = scl_obj.transform(datapoint)
		print(X_train_scaled)
		X_train_scaled=np.array(X_train_scaled)
		#X_train_scaled.reshape(1,-1).astype('float32')
		x=X_train_scaled
		value=preprocessing().predict(x)
		print(Value)
	#return render_template('index.html')
	return Value

if __name__=='__main__' :
	app.secret_key = 'super secret key'

	app.debug = True
	app.run()


