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
Adam = tf.keras.optimizers.Adam

sys.path.append(os.environ['CMS_ROOT'])
from cms_modules.utils import train_valid_split_w_sampling, get_next_run_description, model_summary_to_string
from cms_modules.utils import get_imbalance_description, dict_to_hdf5, dict_from_hdf5, args_to_dict
from cms_modules.plotting import plot_train_vs_validation
from cms_modules.logging import Logger
from cms_modules.threshold_utils import calculate_threshold_scores
from cms_modules.keras_callbacks import CustomMetricsCallback

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

threshold_interval = float(cli_args.get('threshold_interval'))

dropout_rate = float(cli_args.get('dropout_rate', 0.5))

activation = 'relu'
learn_rate = 1e-3
data_path = 'partd-2017-minmax-scaled.hdf5'


# INITIALIZE LOGGER
# -------------------------------------------------- #
logger = Logger()
logger.log_message('Executing ' + filename)
logger.log_message('\n'.join(sys.argv[1:]))


# DEFINE DIRECTORIES/PATHS
# -------------------------------------------------- #
# data
hdf5_path = os.path.join(os.environ['CMS_ROOT'], 'data', data_path)
logger.log_message(hdf5_path)
train_key = 'train_normalized'
test_key = 'test_normalized'
# results
results_dir = './results'
train_results = 'train_metrics.hdf5'
valid_results = 'valid_metrics.hdf5'
# create results dir if DNE
if not os.path.isdir(results_dir):
	os.makedirs(results_dir)


# LOOP OVER TOTAL RUNS, GENERATING RESULTS FOR EACH
# -------------------------------------------------- #
for run in range(runs):

	logger.log_time('BEGINNING RUN ' + str(run) + ': ')

	# resolving keras memory leak
	K.clear_session()
	tf.reset_default_graph()


	# LOAD ALREADY NORMALIZED DATA
	# -------------------------------------------------- #
	data = pd.read_hdf(hdf5_path, key=train_key)
	logger.log_message('Data imbalance levels before sampling')
	logger.log_message(get_imbalance_description(data['exclusion']))


	# CREATE DIRECTORY TO STORE RESULTS
	# -------------------------------------------------- #
	run_desc = get_next_run_description(results_dir)
	run_results_dir = os.path.join(results_dir, run_desc)
	os.mkdir(run_results_dir)
	train_results_path = os.path.join(run_results_dir, train_results)
	valid_results_path = os.path.join(run_results_dir, valid_results)


	# SPLIT INTRO TRAIN / VALIDATION SETS WITH SAMPLING
	# -------------------------------------------------- #
	train_x, train_y, valid_x, valid_y = train_valid_split_w_sampling(
		data,
		valid_size=0.1,
		target_col='exclusion',
		ros_rate=ros_rate,
		rus_rate=rus_rate)

	del data

	logger.log_message('Training imbalance levels after sampling')
	logger.log_message(get_imbalance_description(train_y))
	logger.log_message('Validation imbalance levels after sampling')
	logger.log_message(get_imbalance_description(valid_y))


	# BUILD MODEL
	# -------------------------------------------------- #
	_, input_dim = train_x.shape

	# hidden layers
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


	# SETUP CALLBACKS TO RECORD TRAIN/VALID METRICS
	# -------------------------------------------------- #
	metric_freq = 1
	trainingMetricsCallback = CustomMetricsCallback(train_x, train_y, metric_freq)
	validationMetricsCallback = CustomMetricsCallback(valid_x, valid_y, metric_freq, validation=True)

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
		validation_data=(valid_x, valid_y),
		verbose=0,
		callbacks=[trainingMetricsCallback, validationMetricsCallback],
	)

	logger.log_time('TRAINING COMPLETE')

	# SAVE RESULTS TO HDF5
	# -------------------------------------------------- #
	training_scores = trainingMetricsCallback.get_scores()
	validation_scores = validationMetricsCallback.get_scores()

	# attach loss to dict
	training_scores['loss'] = history.history['loss']
	validation_scores['loss'] = history.history['val_loss']

	# write training and validation metrics to results dir
	dict_to_hdf5(training_scores, train_results_path)
	dict_to_hdf5(validation_scores, valid_results_path)

	# plot training vs validation and write to results dir
	plot_path = os.path.join(run_results_dir, 'train_validation_metrics.png')
	metrics_to_plot = ['roc_auc', 'recall', 'specificity', 'balanced_acc', 'loss']
	logger.log_message('PLOTTING TRAIN VS VALIDATION PERFORMANCE')
	plot_train_vs_validation(metrics_to_plot, training_scores, validation_scores, plot_path)

	# save threshold values
	y_prob = model.predict(valid_x)
	logger.log_message('CALCULATING THRESHOLDS')
	calculate_threshold_scores(valid_y, y_prob, run_results_dir, threshold_interval)

	logger.log_time('RESULTS SAVED')
	logger.log_time('ENDING RUN ' + str(run))
	logger.write_to_file(os.path.join(run_results_dir, 'logs.txt'))

	del train_x
	del train_y
	del valid_x
	del valid_y
	del training_scores
	del validation_scores


logger.log_time('ALL RUNS COMPLETED')
logger.mark_dir_complete('./')
