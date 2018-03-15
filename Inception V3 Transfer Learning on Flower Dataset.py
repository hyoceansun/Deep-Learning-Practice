import tensorflow as tf
import os.path
import random
import numpy as np
from tensorflow.python.platform import gfile
import glob
import random

bottleneck_tensor_size=2048
bottleneck_tensor_name='pool_3/_reshape:0'
jpeg_data_tensor_name='DecodeJpeg/contents:0'
model_dir='C:/Users/User/PycharmProjects/Neural_Network'
model_file='tensorflow_inception_graph.pb'
cache_dir='C:/Users/User/PycharmProjects/Neural_Network/bottleneck'
input_data='C:/Users/User/PycharmProjects/Neural_Network/flower_photos'

validation_percentage=10
test_percentage=10
learning_rate=0.01
steps=4000
batch=100
processing_rate=0.05

def image_preprocessing(image_path):
    image_raw_data=tf.read_file(image_path)   #This is the encoded image, image_raw_data.eval() gives byte
    img_data = tf.image.decode_jpeg(image_raw_data)   #After decoding, img_data.eval()  gives ndarray
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    img_data = tf.image.random_brightness(img_data, max_delta=50 / 255)
    img_data = tf.image.random_saturation(img_data, 0.5, 1.5)
    img_data = tf.image.random_contrast(img_data, 0.5, 1.5)
    img_data_np = sess.run(img_data)
    min_val = np.min(img_data_np)
    max_val = np.max(img_data_np)
    img_data_clamped = (img_data_np - min_val) / (max_val - min_val)
    img_data = tf.convert_to_tensor(img_data_clamped)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
    image_data=sess.run(tf.image.encode_jpeg(img_data))  #encoded image.eval() gives byte
    return image_data


def create_image_lists(testing_percentage,validation_percentage):
    # result is a dictionary containing key (label name) and value, which is also a dictioanry consisting of image names
    result={}
    sub_dirs=[x[0] for x in os.walk(input_data)] #gives sub directory for each type of flowers  e.g. 'C:/Users/User/PycharmProjects/Neural_Network/flower_photos\\daisy'
    is_root_dir=True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False #skip the first directory which is current directory (flower photos)
            continue

        extensions=['jpg','jpeg','JPG','JPEG']
        file_list=[]
        dir_name=os.path.basename(sub_dir) #basename gives daisy, rose etc.
        for extension in extensions:
            file_glob=os.path.join(input_data,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob)) #Get a list of individual image directories
        if not file_list: continue

        label_name=dir_name.lower()
        training_images=[]
        testing_images=[]
        validation_images=[]
        for file_name in file_list:
            base_name=os.path.basename(file_name)  #e.g. '100080576_f52e8ee070_n.jpg'
            chance=np.random.randint(100)
            if chance<validation_percentage:
                validation_images.append(base_name)
            elif chance<(testing_percentage+validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name]={'dir':dir_name,'training':training_images,'testing':testing_images,'validation':validation_images}
    return result

def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists=image_lists[label_name]
    category_list=label_lists[category]
    mod_index=index % len(category_list)
    base_name=category_list[mod_index]
    sub_dir=label_lists['dir']
    full_path=os.path.join(image_dir,sub_dir,base_name)
    return full_path

def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,cache_dir,label_name,index,category)+'.txt'

def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    bottleneck_values=sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    bottleneck_values=np.squeeze(bottleneck_values)
    #final_bottleneck_values=[x[0] for x in bottleneck_values]
    return bottleneck_values

def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    label_lists=image_lists[label_name]
    sub_dir=label_lists['dir']
    sub_dir_path=os.path.join(cache_dir,sub_dir)
    if not os.path.exists(sub_dir_path):os.makedirs(sub_dir_path)
    bottleneck_path=get_bottleneck_path(image_lists,label_name,index,category)


    if not os.path.exists(bottleneck_path):
        image_path=get_image_path(image_lists,input_data,label_name,index,category)
        if random.uniform(0,1)<processing_rate:
            image_data=image_preprocessing(image_path)
        else:
            image_data=gfile.FastGFile(image_path,'rb').read()
        bottleneck_values=run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        bottleneck_string=','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string=bottleneck_file.read()
        bottleneck_values=[float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

def get_random_cached_bottlenecks(sess,n_classes,image_lists,how_many, category, jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    for _ in np.arange(how_many):
        label_index=random.randrange(n_classes)
        label_name=list(image_lists.keys())[label_index]
        image_index=random.randrange(65536)
        bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor)
        ground_truth=np.zeros(n_classes,dtype=np.float32)
        ground_truth[label_index]=1
        bottlenecks.append(bottleneck)
        ground_truths.append(list(ground_truth))
    return bottlenecks,ground_truths

def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    label_name_list=list(image_lists.keys())
    for label_index,label_name in enumerate(label_name_list):
        category='testing'
        for index,unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1
            bottlenecks.append(bottleneck)
            ground_truths.append(list(ground_truth))
    return bottlenecks,ground_truths


image_lists=create_image_lists(test_percentage,validation_percentage)
n_classes=len(image_lists.keys())
with gfile.FastGFile(os.path.join(model_dir,model_file),'rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(graph_def,return_elements=[bottleneck_tensor_name,jpeg_data_tensor_name])
bottleneck_input=tf.placeholder(tf.float32,[None,bottleneck_tensor_size],name='BottleneckInputPlaceholder')
ground_truth_input=tf.placeholder(tf.float32,[None,n_classes],name='GroundTruthInput')
with tf.name_scope('final_training_ops'):
    weights=tf.Variable(tf.truncated_normal([bottleneck_tensor_size,n_classes],stddev=0.001))
    biases=tf.Variable(tf.zeros([n_classes]))
    logits=tf.matmul(bottleneck_input,weights)+biases
    final_tensor=tf.nn.softmax(logits) # n * t

cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=ground_truth_input)
cross_entropy_mean=tf.reduce_mean(cross_entropy)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)
with tf.name_scope('evaluation'):
    correct_prediction=tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
    evaluation_step=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    for i in np.arange(steps):
        train_bottlenecks,train_ground_truth=get_random_cached_bottlenecks(sess,n_classes,image_lists,batch,'training',jpeg_data_tensor,bottleneck_tensor)
        sess.run(train_step,feed_dict={bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truth})

    test_bottlenecks,test_ground_truth=get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor)
    test_accuracy=sess.run(evaluation_step,feed_dict={bottleneck_input:test_bottlenecks,ground_truth_input:test_ground_truth})
    print('Final test accuracy=%.1f%%'% (test_accuracy*100))


