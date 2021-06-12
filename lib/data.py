from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature
import tensorflow as tf
import numpy as np
from io import BytesIO
from tensorflow.python.lib.io import file_io


############################################### Serialsing ###################################################
def transform_states(data):

    ID,pos, dimensions, color, border, fill, text, img, seq_len, seq_mask = data['ID'], data['pos'], data['dimensions'], data['color'], \
                                                data['border'], data['fill'], data['text'], \
                                                tf.io.read_file(data['img_paths']), \
                                                data['seq_lens'], data['seq_masks'] \

    return {
            'ID' : ID,
            'pos' : pos,
            'dimensions' : dimensions,
            'color' : color,
            'border' : border,
            'fill' : fill,
            'text' : text,
            'img': img,
            'seq_len':seq_len,
            'seq_mask':seq_mask,
            }

def transform_dataset(dataset):
  return dataset.map(transform_states, num_parallel_calls=4)


def serialise(data):
    
    ID,pos, dimensions, color, border, fill, text, img, seq_len, seq_mask = data['ID'], data['pos'], data['dimensions'], data['color'], \
                                    data['border'], data['fill'], data['text'], data['img'], \
                                    int(data['seq_len']), data['seq_mask'] \

    ID = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(ID).numpy(),]))
    pos = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(pos).numpy(),]))
    dimensions = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(dimensions).numpy(),]))
    color = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(color).numpy(),]))
    border = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(border).numpy(),]))
    fill = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(fill).numpy(),]))
    text = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(text).numpy(),]))
    img = Feature(bytes_list=BytesList(value=[img.numpy(),]))
    seq_len =  Feature(int64_list=Int64List(value=[seq_len,]))
    seq_mask = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(seq_mask).numpy(),]))
    # img is already serialised because we never decode it!
    
    features = Features(feature={
                'ID' : ID,
                'pos' : pos,
                'dimensions' : dimensions,
                'color' : color,
                'border' : border,
                'fill' : fill,
                'text' : text,
                'img': img,
                'seq_len':seq_len,
                'seq_mask':seq_mask,
                })
    
    example = Example(features=features)
    
    return example.SerializeToString()

###################################################### Deserialising ##############################################

def decode_img(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    #image = tf.reshape(image, [slide_width//IMAGE_RES_DIVISOR, slide_height//IMAGE_RES_DIVISOR, 3]) # explicit size needed for TPU
    return image


def decode_img_prepro_inception(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image



class dataloader():
    def __init__(self,
                paths,
                shuffle_size=64,
                batch_size=512,
                num_devices=1, 
                inception_prepro = False):

        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.prefetch_size = tf.data.experimental.AUTOTUNE
        self.inception_prepro = inception_prepro

        # metadata = src = str(path)+'/metadata.npz'
        # f = BytesIO(file_io.read_file_to_string(src, binary_mode=True))
        # metadata = np.load(f)
        # # Basically envisaging it as some tokens for the img, some for a sentence description, some for the ppt vectors
        # self.img_tokens = metadata['img_tokens']
        # self.language_tokens = metadata['language_tokens']
        # self.component_tokens = metadata['component_tokens']
        

        records = []
        for p in paths:
            records += tf.io.gfile.glob(f"{p}/tf_records/*.tfrecords")

        self.dataset = self.load_tf_records(records)
        self.dataset = (self.dataset
                        .repeat()
                        .shuffle(self.shuffle_size)
                        .batch(self.batch_size, drop_remainder=True)
                        .prefetch(self.prefetch_size))

                    

        if num_devices > 1:
            self.dataset = self.dataset.batch(num_devices)


    def read_tfrecord(self, example):
        LABELED_TFREC_FORMAT = {
                'ID': tf.io.FixedLenFeature([], tf.string),
                'pos': tf.io.FixedLenFeature([], tf.string),
                'dimensions': tf.io.FixedLenFeature([], tf.string),
                'color': tf.io.FixedLenFeature([], tf.string),
                'border': tf.io.FixedLenFeature([], tf.string),
                'fill': tf.io.FixedLenFeature([], tf.string),
                'text': tf.io.FixedLenFeature([], tf.string),
                'img': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
                'seq_len': tf.io.FixedLenFeature([], tf.int64),
                'seq_mask':tf.io.FixedLenFeature([], tf.string),
        }
        data = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        ID = tf.io.parse_tensor(data['ID'], tf.int32) 
        if self.inception_prepro:
            img = decode_img_prepro_inception(data['img'])
        else:
            img = decode_img(data['img'])
        seq_len = tf.cast(data['seq_len'], tf.int32) 
        seq_mask = tf.io.parse_tensor(data['seq_mask'], tf.float32) 

        return {'sequence' : ID[:-1], # all component tokens except for the last - which will be either end or padding
                'target': ID[1:], # all component tokens except for the start token 
                'img' : img,
                'seq_len': seq_len,
                'seq_mask':seq_mask}

    def load_tf_records(self, filenames, ordered=False):
        # Read from TFRecords. For optimal performance, reading from multiple files at once and
        # disregarding data order. Order does not matter since we will be shuffling the data anyway.

        # check, does this ignore intra order or just inter order? Both are an issue!
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False # disable order, increase speed

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4) # automatically interleaves reads from multiple files - keep it at 1 we need the order
        dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset