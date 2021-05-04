from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature
import tensorflow as tf


############################################### Serialsing ###################################################
def transform_states(data):

    vec, img = data['vecs'], tf.io.read_file(data['img_paths'])

    return {
            'vec': vec,
            'img': img,
            }

def transform_dataset(dataset):
  return dataset.map(transform_states, num_parallel_calls=4)


def serialise(data):
    
    vec, img = data['vec'], data['img']
    
    vec = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(vec).numpy(),]))
    img = Feature(bytes_list=BytesList(value=[img.numpy(),]))
    # img is already serialised because we never decode it!
    
    features = Features(feature={
                'vec': vec,
                'img': img,
                })
    
    example = Example(features=features)
    
    return example.SerializeToString()

###################################################### Deserialising ##############################################

def decode_img(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    #image = tf.reshape(image, [slide_width//IMAGE_RES_DIVISOR, slide_height//IMAGE_RES_DIVISOR, 3]) # explicit size needed for TPU
    return image



def read_tfrecord(example):
    LABELED_TFREC_FORMAT = {
            'vec': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'img': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
    }
    data = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    
    vec = tf.io.parse_tensor(data['vec'], tf.float32) 
    img = decode_img(data['img'])

    return {'vec' : vec, 
            'img' : img}

def load_tf_records(filenames, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    # check, does this ignore intra order or just inter order? Both are an issue!
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4) # automatically interleaves reads from multiple files - keep it at 1 we need the order
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


class dataloader():
    def __init__(self,
                path,
                shuffle_size=64,
                batch_size=512):

        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.prefetch_size = tf.data.experimental.AUTOTUNE

        records = tf.io.gfile.glob(f"{path}\\tf_records/*.tfrecords")
        self.dataset = load_tf_records(records)
        self.dataset = (self.dataset
                        .repeat()
                        .shuffle(self.shuffle_size)
                        .batch(self.batch_size, drop_remainder=True)
                        .prefetch(self.prefetch_size))