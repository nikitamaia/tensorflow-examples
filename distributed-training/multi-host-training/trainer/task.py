import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import json
import os

from trainer.model import create_model

PER_WORKER_BATCH_SIZE = 64

def get_args():
  '''Parses args.'''
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--epochs',
      required=True,
      type=int,
      help='number training epochs')
  parser.add_argument(
      '--job-dir',
      required=True,
      type=str,
      help='dir to save model')
  args = parser.parse_args()
  return args


def preprocess_data(image, label):
  '''Resizes and scales images.'''

  image = tf.image.resize(image, (300,300))
  return tf.cast(image, tf.float32) / 255., label


def get_batch_size():
  '''Computes global batch size.'''

  tf_config = json.loads(os.environ['TF_CONFIG'])
  num_workers = len(tf_config['cluster']['worker'])
  global_batch_size = PER_WORKER_BATCH_SIZE * num_workers
  return global_batch_size

def create_dataset():
  '''Load Cassava dataset and preprocess data.'''
  
  batch_size = get_batch_size()
  data, info = tfds.load(name='cassava', as_supervised=True, with_info=True)
  number_of_classes = info.features['label'].num_classes
  train_data = data['train'].map(preprocess_data, 
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_data  = train_data.shuffle(1000)
  train_data  = train_data.batch(batch_size)
  train_data  = train_data.prefetch(tf.data.experimental.AUTOTUNE)
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  train_data = train_data.with_options(options) 
  return train_data, number_of_classes


def _is_chief(task_type, task_id):
  '''Determines of machine is chief.'''

  # If `task_type` is None, this may be operating as single worker, which  
  # works effectively as chief.
  return task_type is None or task_type == 'chief' or (
            task_type == 'worker' and task_id == 0)


def _get_temp_dir(dirpath, task_id):
  '''Gets temporary directory for saving model.'''

  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir


def write_filepath(filepath, task_type, task_id):
  '''Gets filepath to save model.'''

  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

def main():
  args = get_args()
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  
  train_data, number_of_classes = create_dataset()

  with strategy.scope():
    model = create_model(number_of_classes)
  model.fit(train_data, epochs=args.epochs)

  # Determine type and task of the machine from 
  # the strategy cluster resolver
  task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id)

  # Based on the type and task, write to the desired model path 
  write_model_path = write_filepath(args.job_dir, task_type, task_id)
  model.save(write_model_path)

if __name__ == "__main__":
    main()
