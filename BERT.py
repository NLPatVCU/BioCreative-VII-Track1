#!/usr/bin/env python
# coding: utf-8
#Author Darshini Mahendran

import os
import shutil
import fnmatch
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from datetime import datetime
from sklearn.model_selection import train_test_split

print("tensorflow version : ", tf.__version__)
print("tensorflow_hub version : ", hub.__version__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True) 
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

shutil.rmtree('drugprot/output/eval')
filelist = [f for f in os.listdir('drugprot/output')]
for f in filelist:
    os.remove(os.path.join('drugprot/output', f))

def read_from_file(file, read_as_int=False):
    """
    Reads a file and returns its contents as a list
    :param read_as_int: read as integer instead of strings
    :param file: path to file to be read
    """

    if not os.path.isfile(file):
        raise FileNotFoundError("Not a valid file path")

    #load as numpy files
    if read_as_int:
        content = np.loadtxt(file, dtype='int')
        content = content.reshape((-1, 3))
    else:
        with open(file) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
    return content

# read input files
track_train = np.asarray(read_from_file('drugprot/track_train')).reshape((-1, 3))
track_test = np.asarray(read_from_file('drugprot/track_test')).reshape((-1, 3))
df_train_sentence = pd.read_csv('drugprot/input_train.txt', header=None)
test = pd.read_csv('drugprot/input_test.txt', header=None)

df_train_label = pd.read_csv('drugprot/labels_train', header=None)
train_org = pd.concat([df_train_sentence, df_train_label], axis=1)
train_org.columns = ['sentence', 'original_label']
test['label'] = np.random.randint(0,13, size=len(test))
test.columns = ['sentence', 'original_label']

train_track = pd.DataFrame()
test_track = pd.DataFrame()
train_track['track'] = track_train.tolist()
test_track['track'] = track_test.tolist()
train_org.reset_index(inplace=True)
test.reset_index(inplace=True)

# create a dictionary
possible_labels = train_org.original_label.unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

train_org['label'] = train_org.original_label.replace(label_dict)
train = train_org.drop(['original_label','index'], axis =1)
test = test.drop(['index'], axis =1)

#split train and validation
train, val =  train_test_split(train, test_size = 0.1, random_state = 100)
print("Training Set Shape :", train.shape)
print("Validation Set Shape :", val.shape)
print("Test Set Shape :", test.shape)


DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'label'
label_list = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13]

train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

val_InputExamples = val.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)


#BERT model
# path to an cased version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

#set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128

# Convert our train and validation features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
val_features = bert.run_classifier.convert_examples_to_features(val_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

#Creating A Multi-Class Classifier Model
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  
  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)

#A function that adapts our model to work for training, evaluation, and prediction.

# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        
        return {
            "eval_accuracy": accuracy,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
            }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn

# Set the output directory for saving model file
OUTPUT_DIR = 'drugprot/output/'

#@markdown Whether or not to clear/delete the directory and create a new one
DO_DELETE = False #@param {type:"boolean"}

if DO_DELETE:
  try:
    tf.gfile.DeleteRecursively(OUTPUT_DIR)
  except:
    pass

tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

# Compute train and warmup steps from batch size
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 15
# Warmup is a period of time where the learning rate is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 300
SAVE_SUMMARY_STEPS = 100

# Compute train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify output directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

# Specify output directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

#Initializing the model and the estimator
model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

# Create an input function for validating. drop_remainder = True for using TPUs.
val_input_fn = run_classifier.input_fn_builder(
    features=val_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

#Training & Evaluating
#Training the model
print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

#Evaluating the model with Validation set
estimator.evaluate(input_fn=val_input_fn, steps=None)

#Predicting For Test Set
# A method to get predictions
def getPrediction(in_sentences):
#A list to map the actual labels to the predictions
  labels = ["ACTIVATOR", "INHIBITOR","AGONIST","ANTAGONIST","AGONIST-INHIBITOR","AGONIST-ACTIVATOR","INDIRECT-DOWNREGULATOR","INDIRECT-UPREGULATOR","DIRECT-REGULATOR","PRODUCT-OF","SUBSTRATE","SUBSTRATE_PRODUCT-OF","PART-OF","No-Relation"]
 
  #Transforming the test data into BERT accepted form
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] 
  
  #Creating input features for Test data
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

  #Predicting the classes 
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  return [(sentence, prediction['probabilities'],prediction['labels'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

# predict
pred_sentences = list(test['sentence'])
predictions = getPrediction(pred_sentences)

enc_labels = []
act_labels = []
for i in range(len(predictions)):
  enc_labels.append(predictions[i][2])
  act_labels.append(predictions[i][3])

initial_predictions = 'drugprot/predictions/initial/'
final_predictions = 'drugprot/predictions/final/'

# flag to write the no-relation into the output files
No_Rel= False

# Delete all files in the folder before the prediction
ext =".ann"
filelist = [f for f in os.listdir(initial_predictions) if f.endswith(ext)]
for f in filelist:
    os.remove(os.path.join(initial_predictions, f))

for  x in range(0, len(act_labels)):
    
    file = test_track.loc[x].tolist()[0][0]
    e1 = test_track.loc[x].tolist()[0][1]
    e2 = test_track.loc[x].tolist()[0][2]
    #check the length of file name and padding
    if len(str(file)) == 1:
        f = "000"+str(file) + ".ann"
    elif len(str(file)) == 2:
        f = "00"+str(file) + ".ann"
    elif len(str(file)) == 3:
        f = "0"+str(file) + ".ann"
    else:
        f = str(file) + ".ann"
    #key for relations (not for a document but for all files)
    key = "R" + str(x + 1)
    # entity pair in the relations
    e1 = "T"+str(e1)
    e2 = "T"+str(e2)
    f1 = open(initial_predictions + str(f), "a")
    label = list(label_dict.keys())[list(label_dict.values()).index(enc_labels[x])]
    if No_Rel:
        #open and append relation the respective files in BRAT format
        f1.write(str(key) + '\t' + str(label) + ' ' + 'Arg1:' + str(e1) + ' ' + 'Arg2:' + str(e2) + '\n')
        f1.close()
    else:
        if label != 'No-Relation':
            # open and append relation the respective files in BRAT format
            f1.write(str(key) + '\t' + str(label) + ' ' + 'Arg1:' + str(e1) + ' ' + 'Arg2:' + str(e2) + '\n')
            f1.close()

# write the predictions to the BRAT format files
def write_entities(input_folder, output_folder):
    # Delete all files in the folder initially to prevent appending
    ext = ".ann"
    filelist = [f for f in os.listdir(output_folder) if f.endswith(ext)]
    for f in filelist:
        os.remove(os.path.join(output_folder, f))

    for f in os.listdir(input_folder):
        if fnmatch.fnmatch(f, '*.ann'):
            print(f)
            annotations = {'entities': {}}
            with open(input_folder + str(f), 'r') as file:
                annotation_text = file.read()

            for line in annotation_text.split("\n"):
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                if "\t" not in line:
                    raise InvalidAnnotationError(
                        "Line chunks in ANN files are separated by tabs, see BRAT guidelines. %s"
                        % line)
                line = line.split("\t")
                if 'T' == line[0][0]:
                    tags = line[1].split(" ")
                    entity_name = tags[0]
                    entity_start = int(tags[1])
                    entity_end = int(tags[-1])
                    annotations['entities'][line[0]] = (entity_name, entity_start, entity_end, line[-1])

            f = open(output_folder + str(f), "a")
            for key in annotations['entities']:
                for label, start, end, context in [annotations['entities'][key]]:
                    f.write(
                        str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')
            f.close()

def append(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        print(filename)
        if os.stat(input_folder + filename).st_size == 0:
            continue
        else:
            df = pd.read_csv(input_folder + filename, header = None, sep="\t")
            df.columns = ['key', 'body']
            df['key'] = df.index
            df['key'] = 'R' + df['key'].astype(str)
            df.to_csv(output_folder  + filename, sep='\t', index=False, header=False, mode='a')

# append and renumber the relations in the output files
append(initial_predictions, final_predictions)



