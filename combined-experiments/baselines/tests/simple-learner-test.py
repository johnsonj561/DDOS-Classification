import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
K = tf.keras.backend
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation
Regularizers = tf.keras.regularizers
Adam = tf.keras.optimizers.Adam

sys.path.append(os.environ['CMS_ROOT'])
from cms_modules.utils import (
	model_summary_to_string,
	apply_ros_rus,
	get_imbalance_description,
	args_to_dict,
	split_on_binary_attribute)
from cms_modules.logging import Logger
from cms_modules.utils import write_performance_metrics
from cms_modules.keras_callbacks import EpochTimerCallback


# PARSE CLI ARGUMENTS
# -------------------------------------------------- #
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)

ros_rate = cli_args.get('ros_rate')
if ros_rate != None:
          ros_rate = float(ros_rate)
rus_rate = cli_args.get('rus_rate')
if rus_rate != None:
	rus_rate = float(rus_rate)

width = int(cli_args.get('width'))
depth = int(cli_args.get('depth'))

epochs = int(cli_args.get('epochs'))
runs = int(cli_args.get('runs'))

decision_threshold = float(cli_args.get('decision_threshold'))
default_threshold = 0.5
theoretical_threshold = "tbd"

minority_size = "tbd"

activation = 'relu'
dropout_rate = 0.5
learn_rate = 1e-3

# INITIALIZE LOGGER
# -------------------------------------------------- #
logger = Logger()
logger.log_message('Executing ' + filename)
logger.log_message('\n'.join(sys.argv[1:]))

# DEFINE DIRECTORIES/PATHS
# -------------------------------------------------- #
data_file = 'combined-minmax-scaled.hdf5'
data_path = '/home/jjohn273/git/DDOS-Classification/data/combined-minmax-scaled.hdf5'
logger.log_message(data_path)
train_key = 'train_normalized'
test_key = 'test_normalized'


# DEFINE THRESHOLDS TO COMPUTE SCORES FOR
# -------------------------------------------------- #
theoretical_threshold_results_file = './theoretical-results.csv'
optimal_threshold_results_file = './optimal-results.csv'
default_threshold_results_file = './default-results.csv'
timings_results_file = './timings.csv'


# LOOP OVER TOTAL RUNS, GENERATING RESULTS FOR EACH
# -------------------------------------------------- #
for run in range(runs):

	# resolving keras memory leak
	K.clear_session()
	tf.reset_default_graph()

	logger.log_time('BEGINNING RUN ' + str(run) + ': ')

	# LOAD NORMALIZED TRAINING DATA
	# -------------------------------------------------- #
	train_data = pd.read_hdf(data_path, key=train_key)
	logger.log_message('Data imbalance levels before sampling')
	logger.log_message(get_imbalance_description(train_data['class']))
	logger.log_message('Size of train data = ' + str(len(train_data)))


	# LOAD NORMALIZED TEST DATA
	# -------------------------------------------------- #
	test_data = pd.read_hdf(data_path, key=test_key)


	# APPLY SAMPLING TO THE TRAINING DATA
	# --------------------------------------------------
	pos_train, neg_train = split_on_binary_attribute(train_data, attribute='class', pos_label=1, neg_label=0)
	train_data = apply_ros_rus(pos_train, neg_train, ros_rate=ros_rate, rus_rate=rus_rate)
	del pos_train
	del neg_train

	# SEPARATE FEATURES/LABELS
	# --------------------------------------------------
	train_y = train_data['class']
	train_x = train_data.drop(columns=['class'])

	test_y = test_data['class']
	test_x = test_data.drop(columns=['class'])

	logger.log_message('Training data imbalance levels after sampling')
	logger.log_message(get_imbalance_description(train_y))
	logger.log_message('Test data imbalance levels')
	logger.log_message(get_imbalance_description(test_y))


	# SEPARATE FEATURES/LABELS
	# --------------------------------------------------
	if theoretical_threshold == "tbd" or minority_size == "tbd":
		pos_length, neg_length = len(train_data.loc[train_data['class'] == 1]), len(train_data.loc[train_data['class'] == 0])
		print(f'Positive lenth {pos_length} and negative lenth {neg_length}')
		train_total = pos_length + neg_length
		print(f'Total is {train_total}')
		theoretical_threshold = pos_length / train_total
		minority_size = theoretical_threshold
		logger.log_message(f'Theoretical threshold is {theoretical_threshold}')
		logger.log_message(f'Minority ratio is {minority_size}')


	del train_data
	del test_data


	# RECORD EPOCH TIMES
	# -------------------------------------------------- #
	epochTimer = EpochTimerCallback(timings_results_file)


	# BUILD MODEL
	# -------------------------------------------------- #
	_, input_dim = train_x.shape

	model = Sequential()
	model.add(Dense(width, input_dim=input_dim))
	model.add(BatchNormalization())
	model.add(Activation(activation))
	model.add(Dropout(dropout_rate))

	for _ in range(depth - 1):
		model.add(Dense(width))
		model.add(BatchNormalization())
		model.add(Activation(activation))
		model.add(Dropout(dropout_rate))

	# classification layer
	model.add(Dense(1, activation='sigmoid'))

	logger.log_message(model_summary_to_string(model))


	# TRAIN MODEL
	# -------------------------------------------------- #
	optimizer = Adam(lr=learn_rate)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

	logger.log_time('MODEL COMPILED')
	logger.log_time('BEGINNING TRAINING')

	history = model.fit(
		x=train_x,
		y=train_y,
		epochs=epochs,
		batch_size=256,
		verbose=0,
		callbacks=[epochTimer],
	)

	logger.log_time('TRAINING COMPLETE')

	# CALCULATE TEST SET PERFORMANCE AND WRITE TO OUT FILE
	# -------------------------------------------------- #
	y_prob = model.predict(test_x)
	write_performance_metrics(test_y, y_prob, theoretical_threshold, theoretical_threshold_results_file, minority_size)
	write_performance_metrics(test_y, y_prob, decision_threshold, optimal_threshold_results_file, minority_size)
	write_performance_metrics(test_y, y_prob, 0.5, default_threshold_results_file, minority_size)
	epochTimer.write_timings()

	logger.log_time('RESULTS SAVED')
	logger.log_time('ENDING RUN ' + str(run))


logger.log_time('ALL RUNS COMPLETED')
logger.write_to_file('logs.txt')
logger.mark_dir_complete('./')
