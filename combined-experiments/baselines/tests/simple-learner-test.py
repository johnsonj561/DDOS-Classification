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


# PARSE CLI ARGUMENTS
# -------------------------------------------------- #
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)

width = int(cli_args.get('width'))
depth = int(cli_args.get('depth'))

epochs = int(cli_args.get('epochs'))
runs = int(cli_args.get('runs'))

decision_threshold = float(cli_args.get('decision_threshold'))

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
# data
data_file = 'combined-minmax-scaled.hdf5'
data_path = os.path.join(os.environ['CMS_ROOT'], 'data', data_file)
logger.log_message(data_path)
train_key = 'train_normalized'
test_key = 'test_normalized'
optimal_threshold_results_file = './optimal-results.csv'
default_threshold_results_file = './default-results.csv'


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
	logger.log_message('Size of train data = ' + str(len(train_data)))


	# LOAD NORMALIZED TEST DATA
	# -------------------------------------------------- #
	test_data = pd.read_hdf(data_path, key=test_key)


	# SEPARATE FEATURES/LABELS
	# --------------------------------------------------
	train_y = train_data['class']
	train_x = train_data.drop(columns=['class'])

	test_y = test_data['class']
	test_x = test_data.drop(columns=['class'])

	del train_data
	del test_data

	logger.log_message('Training data imbalance levels after sampling')
	logger.log_message(get_imbalance_description(train_y))
	logger.log_message('Test data imbalance levels')
	logger.log_message(get_imbalance_description(test_y))


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
	)

	logger.log_time('TRAINING COMPLETE')

	# CALCULATE TEST SET PERFORMANCE AND WRITE TO OUT FILE
	# -------------------------------------------------- #
	y_prob = model.predict(test_x)
	write_performance_metrics(test_y, y_prob, decision_threshold, optimal_threshold_results_file)
	write_performance_metrics(test_y, y_prob, 0.5, default_threshold_results_file)

	# TODO
	# add average training time and minority ratio to the results

	logger.log_time('RESULTS SAVED')
	logger.log_time('ENDING RUN ' + str(run))


logger.log_time('ALL RUNS COMPLETED')
logger.write_to_file('logs.txt')
logger.mark_dir_complete('./')
