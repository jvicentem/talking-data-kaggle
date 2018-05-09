import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import StratifiedKFold
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def data_for_model(data_x):
	aux = data_x[['click_time_hour']]

	one_hot_encoding = OneHotEncoder().fit(aux).transform(aux)	

	aux = data_x[['device_custom_score', 'os_custom_score', 'app_custom_score', 'channel_custom_score', 'n_previous_clicks', 'click_time_diff']]

	data_x = MinMaxScaler().fit(aux).transform(aux)	

	return np.hstack((data_x, one_hot_encoding.toarray()))

def balance_train(train_x, train_y, seed):
	sru  =  RandomUnderSampler(random_state=seed, ratio='auto') 

	train_x, train_y = sru.fit_sample(train_x, train_y)
	

def plot_var_importance(model):
	feature_names = ['device_custom_score', 'os_custom_score', 'app_custom_score', 'channel_custom_score', 
	                 'click_time_hour.0', 
	                 'click_time_hour.1', 
	                 'click_time_hour.2',
	                 'click_time_hour.3',
	                 'click_time_hour.4',
	                 'click_time_hour.5',
	                 'click_time_hour.6',
	                 'click_time_hour.7',
	                 'click_time_hour.8',
	                 'click_time_hour.9',
	                 'click_time_hour.10',
	                 'click_time_hour.11',
	                 'click_time_hour.12',
	                 'click_time_hour.13',
	                 'click_time_hour.14',
	                 'click_time_hour.15',
	                 'click_time_hour.16',
	                 'click_time_hour.17',
	                 'click_time_hour.18',
	                 'click_time_hour.19',
	                 'click_time_hour.20',
	                 'click_time_hour.21',
	                 'click_time_hour.22',
	                 'click_time_hour.23']

	skplt.plot_feature_importances(model, feature_names=feature_names, max_num_features=10, figsize=(50, 10))
	plt.show()	

'''
Performance report for categorical data 
given an array of real values and an array of predicted values.
'''
def predicted_report(y_test, y_pred):
	results_to_vals = np.vectorize(lambda x: '1' if x == 1 else '0')
	y_test_str = results_to_vals(y_test)
	y_pref_str = results_to_vals(y_pred)
	print('%s\n' % pd.crosstab(y_test_str, y_pref_str, rownames=['Actual'], colnames=['Predicted'], margins=True))
	print(classification_report(y_test_str, y_pref_str))

def report_and_roc_plot(data_x, data_y, model):
	results_to_vals = np.vectorize(lambda x: '1' if x == 1 else '0')

	predicted_report(data_y, model.predict(data_x))
	skplt.plot_roc_curve(results_to_vals(data_y), model.predict_proba(data_x))
	plt.show()
