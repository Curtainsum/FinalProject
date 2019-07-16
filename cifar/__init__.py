import numpy as np
from io import BytesIO
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

ckpt_dir = "../cifar/CIFAR10-log/"
graph = tf.Graph()
sess = tf.Session(graph=graph)


def weight(shape):  # 定义权值
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')


def bias(shape):  # 定义偏置
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')


def conv2d(x, W):  # 定义卷积操作，步长为1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):  # 定义池化，步长为2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with graph.as_default():
    with tf.name_scope('input_layer'):  # 输入层
        x = tf.placeholder('float', shape=[None, 32, 32, 3], name='x')
    with tf.name_scope('conv_1'):  # 卷积层1
        W1 = weight([3, 3, 3, 32])
        b1 = bias([32])
        conv_1 = conv2d(x, W1) + b1
        conv_1 = tf.nn.relu(conv_1)
    with tf.name_scope('pool_1'):  # 池化层1
        pool_1 = max_pool_2x2(conv_1)
    with tf.name_scope('conv_2'):  # 卷积层2
        W2 = weight([3, 3, 32, 64])
        b2 = bias([64])
        conv_2 = conv2d(pool_1, W2) + b2
        conv_2 = tf.nn.relu(conv_2)
    with tf.name_scope('pool_2'):  # 池化层2
        pool_2 = max_pool_2x2(conv_2)
    with tf.name_scope('fc'):  # 全连接层
        W3 = weight([4096, 256])
        b3 = bias([256])
        flat = tf.reshape(pool_2, [-1, 4096])
        h = tf.nn.relu(tf.matmul(flat, W3) + b3)
        h_dropout = tf.nn.dropout(h, keep_prob=0.8)
    with tf.name_scope('output_layer'):  # 输出层
        W4 = weight([256, 10])
        b4 = bias([10])
        pred = tf.nn.softmax(tf.matmul(h_dropout, W4) + b4)
    with tf.name_scope('optimizer'):
        y = tf.placeholder('float', shape=[None, 10], name='label')
        loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)
    with tf.name_scope("evaluation"):  # 准确率
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    ckpt = tf.train.latest_checkpoint(ckpt_dir)  # 检查点文件
    saver = tf.train.Saver(max_to_keep=1)
    saver.restore(sess, ckpt)  # 加载所有参数

label_type1_dict = {0: "载具", 1: "载具", 2: "动物", 3: "动物", 4: "动物", 5: "动物", 6: "动物", 7: "动物", 8: "载具", 9: "载具"}
label_type2_dict = {0: "飞机", 1: "汽车", 2: "鸟", 3: "猫", 4: "鹿", 5: "狗", 6: "蛤", 7: "🐴", 8: "船", 9: "卡车"}


def draw(images):
    plt.imshow(images, cmap='binary')
    plt.show()


print("restore")


def predict(img_data):
    try:
        global sess
        global graph
        global pred

        with graph.as_default():
            img = Image.open(BytesIO(img_data))
            img = img.resize((32, 32)).convert("RGB")
            img_array = np.array([np.array(img).astype('float32') / 255.0])

            # draw(img_array[0])  # 即时绘制

            test = sess.run(pred, feed_dict={x: img_array})
            prediction_rst = sess.run(tf.argmax(test, 1))
        return True, label_type1_dict[prediction_rst[0]], label_type2_dict[prediction_rst[0]]
    except Exception as e:
        print(str(e))
        return False, None, None


init_file = open(r"../cifar/init_test.jpg", "rb")
predict(init_file.read())
init_file.close()

print("init")
