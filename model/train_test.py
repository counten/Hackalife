import time

import tensorflow as tf

from back.configs import Configs
from back.util.fetch_data import data_augmentation, fetch_data, color_pre_process, image_size, class_num

conf = Configs()
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_save_path', conf.dir_log + '/test_log', 'Directory where to save tensorboard log')
tf.app.flags.DEFINE_string('model_save_path', conf.dir_model, 'Directory where to save model weights')
tf.app.flags.DEFINE_string('dir_train_file', conf.dir_train_file, 'Directory where to save train file')
# tf.app.flags.DEFINE_string('ckpt_save_path', './checkpt', 'Directory where to save checkpoint')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('iteration', 1, 'iteration')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')
tf.app.flags.DEFINE_float('epochs', 2, 'epochs')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')


# 卷积
def conv(x, is_train, shape):
    he_initializer = tf.contrib.keras.initializers.he_normal()
    W = tf.get_variable('weights', shape=shape, initializer=he_initializer)
    b = tf.get_variable('bias', shape=[shape[3]], initializer=tf.zeros_initializer)
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=is_train,
                                        updates_collections=None)


# 激活函数
def activation(x):
    return tf.nn.relu(x)


def max_pool(input, k_size=3, stride=2):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME')


def global_avg_pool(input, k_size=1, stride=1):
    return tf.nn.avg_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='VALID')


# 学习率控制
def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.085
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001


def main(_):
    train_x, train_y, test_x, test_y = fetch_data(FLAGS.dir_train_file)
    train_x, test_x = color_pre_process(train_x, test_x)

    # 模型使用三层卷积，命名为conv1, conv2, conv3,
    # 每层卷积后创建2层池化层，命名为mlp1 - 1， mlp1 - 2，
    # 再进行最大池化、dropout防止过拟合后返回一层输出层并作为下一层的输入。
    # 最后第三层卷积后的输出作为softmax层的输入，返回图像数据与label数组。
    # define placeholder x, y_ , keep_prob, learning_rate
    input_x = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='input_x')
    input_y = tf.placeholder(tf.float32, [None, class_num], name='input_y')
    use_bn = tf.placeholder(tf.bool, name='use_bn')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # 搭建模型
    with tf.variable_scope('conv1'):
        output = conv(input_x, use_bn, [5, 5, 3, 192])
        output = activation(output)

    with tf.variable_scope('mlp1-1'):
        output = conv(output, use_bn, [1, 1, 192, 160])
        output = activation(output)

    with tf.variable_scope('mlp1-2'):
        output = conv(output, use_bn, [1, 1, 160, 96])
        output = activation(output)

    with tf.name_scope('max_pool-1'):
        output = max_pool(output, 3, 2)

    with tf.name_scope('dropout-1'):
        output = tf.nn.dropout(output, keep_prob)

    with tf.variable_scope('conv2'):
        output = conv(output, use_bn, [5, 5, 96, 192])
        output = activation(output)

    with tf.variable_scope('mlp2-1'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp2-2'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)

    with tf.name_scope('max_pool-2'):
        output = max_pool(output, 3, 2)

    with tf.name_scope('dropout-2'):
        output = tf.nn.dropout(output, keep_prob)

    with tf.variable_scope('conv3'):
        output = conv(output, use_bn, [3, 3, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp3-1'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp3-2'):
        output = conv(output, use_bn, [1, 1, 192, 2])
        output = activation(output)

    with tf.name_scope('global_avg_pool'):
        output = global_avg_pool(output, 8, 1)

    # with tf.name_scope('softmax'):
    model = tf.reshape(output, [-1, 2], name='model')

    # 模型创建好后将交叉熵，损失精度，训练步数和预测等变量加入tensor：
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=model))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    # 使用Momentum 作为迭代优化器
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum, use_nesterov=True).minimize(
        cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 然后定义saver变量作为Saver函数的调用保存model：
    # saver = tf.train.Saver()

    # for testing
    def run_testing(sess):
        acc = 0.0
        loss = 0.0
        pre_index = 0
        add = 30
        for it in range(10):
            batch_x = test_x[pre_index:pre_index + add]
            batch_y = test_y[pre_index:pre_index + add]
            pre_index = pre_index + add
            loss_, acc_ = sess.run([cross_entropy, accuracy],
                                   feed_dict={input_x: batch_x, input_y: batch_y, use_bn: False, keep_prob: 1.0})
            loss += loss_ / 10.0
            acc += acc_ / 10.0
        summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
                                    tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
        return acc, loss, summary

    # 模型搭建完毕后创建一个session对模型进行训练和测试，
    # 使用flag变量作为参数，
    # 当迭代次数一定后保存模型并输出当前模型的预测精度：
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "./check/model.ckpt")
        summary_writer = tf.summary.FileWriter(FLAGS.log_save_path, sess.graph)

        for ep in range(1, int(FLAGS.epochs) + 1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\n epoch %d/%d:" % (ep, FLAGS.epochs))

            for it in range(1, FLAGS.iteration + 1):
                if pre_index + FLAGS.batch_size < 50000:
                    batch_x = train_x[pre_index:pre_index + FLAGS.batch_size]
                    batch_y = train_y[pre_index:pre_index + FLAGS.batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]

                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={input_x: batch_x, input_y: batch_y, use_bn: True,
                                                    keep_prob: FLAGS.dropout,
                                                    learning_rate: lr})
                batch_acc = accuracy.eval(feed_dict={input_x: batch_x, input_y: batch_y, use_bn: True, keep_prob: 1.0})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += FLAGS.batch_size
                # 测试数据集的评估
                if it == FLAGS.iteration:
                    train_loss /= FLAGS.iteration
                    train_acc /= FLAGS.iteration

                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                      tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc, val_loss, test_summary = run_testing(sess)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print(
                        "iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                        "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                        % (
                        it, FLAGS.iteration, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                    # checkpt_path = "./check/model.ckpt"
                    # saver.save(sess, checkpt_path)
                    # print("Model saved in file: %s" % save_path)
                else:
                    # 训练数据集的评估
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" % (
                        it, FLAGS.iteration, train_loss / it, train_acc / it), end='\r')
                    # checkpt_path = "./check/model.ckpt"
                    # saver.save(sess, checkpt_path)

        # save_path = saver.save(sess, FLAGS.model_save_path)
        # print("Model saved in file: %s" % save_path)
        tf.saved_model.simple_save(sess, FLAGS.model_save_path,
                                   inputs={"input_x": input_x, "input_y": input_y, "use_bn": use_bn},
                                   outputs={"model": model})
        print("Model saved in file: %s" % FLAGS.model_save_path)


# 在经过82次模型的运行，
# 使用0.01的learning rate，
# 每次进行43次迭代后可以看到该模型预测精度可以达到97%：
if __name__ == '__main__':
    tf.app.run()
