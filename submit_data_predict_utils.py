from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd 
import numpy as np

def weekday_number(wday_name):
    if 'Mon' == wday_name:
        return 1
    elif 'Tue' == wday_name:
        return 2
    elif 'Wed' == wday_name:
        return 3
    elif 'Thu' == wday_name:
        return 4
    elif 'Fri' == wday_name:
        return 5
    elif 'Sat' == wday_name:
        return 6
    elif 'Sun' == wday_name:
        return 7

def model_data_preprocessing_for_submit_data(submit_df):
	aux_all_hours = pd.DataFrame([  [0, 0 , 0, 0, 'Fri', 0, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 1, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 2, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 3, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 4, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 5, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 6, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 7, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 8, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 9, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 10, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 11, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 12, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 13, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 14, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 15, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 16, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 17, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 18, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 19, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 20, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 21, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 22, 0 , 0],
									[0, 0 , 0, 0, 'Fri', 23, 0 , 0]
								 ], 
								columns=['os_custom_score','app_custom_score','channel_custom_score','device_custom_score','click_time_wday','click_time_hour','n_previous_clicks','click_time_diff'])

	submit_df = submit_df[['device_custom_score', 'os_custom_score','app_custom_score','channel_custom_score', 'n_previous_clicks', 'click_time_diff', 'click_time_hour']].append(aux_all_hours[['device_custom_score', 'os_custom_score','app_custom_score','channel_custom_score', 'n_previous_clicks', 'click_time_diff', 'click_time_hour']])

	aux = submit_df[['click_time_hour']]

	one_hot_encoding = OneHotEncoder().fit(aux).transform(aux)

	aux = submit_df[['device_custom_score', 'os_custom_score', 'app_custom_score', 'channel_custom_score', 'n_previous_clicks', 'click_time_diff']]

	submit_df = MinMaxScaler().fit(aux).transform(aux)	

	submit_df = np.hstack((submit_df, one_hot_encoding.toarray()))

	return submit_df