import pickle
import random
import numpy

class_num = 2
image_size = 32
image_channels = 3


# 从文件中读取所有数据【label，image_data】
def unpickle(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


# 分离文件中的label和data
def load_data_once(filename):
    batch = unpickle(filename)
    data = batch['data']
    labels = batch['label']
    print("reading data and labels from %s" % filename)
    return data, labels


def load_data(filequeue, data_dir, labels_count):
    global image_size, image_channels

    data, labels = load_data_once(data_dir + '/' + filequeue[0])
    for f in filequeue[1:]:
        data_f, label_f = load_data_once(data_dir + '/' + f)
        data = numpy.append(data, data_f, axis=0)
        labels = numpy.append(labels, label_f, axis=0)
    labels = numpy.array([[float(i == label) for i in range(labels_count)]
                          for label in labels])
    data = data.reshape([-1, image_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


# 从相应文件中返回数据和标签
def fetch_data(data_dir):
    # data_dir = "data/train_file"
    img_depth = image_size * image_size * image_channels

    meta = unpickle(data_dir + '/batches.meta')
    labels_name = meta[b'label_names']
    label_count = len(labels_name)
    # train_files = [ 'data_batch_%d ' % d for d in range(1,6) ]

    train_data, train_labels = load_data(['data_batch_train'], data_dir, label_count)
    test_data, test_labels = load_data(['data_batch_test'], data_dir, label_count)

    print("~~~~~~Reading Data End~~~~~~")
    print("Train data: ", numpy.shape(train_data), numpy.shape(train_labels))
    print("Test data: ", numpy.shape(test_data), numpy.shape(test_labels))

    print("~~~~~~Shuffling Data Start~~~~~~")

    index = numpy.random.permutation(len(train_data))
    train_data = train_data[index]
    train_labels = train_labels[index]

    return train_data, train_labels, test_data, test_labels


# 随机截取图像
def random_crop(batch, crop_shape, padding=None):
    img_shape = numpy.shape(batch[0])

    if padding:
        img_shape = (img_shape[0] + 2 * padding, img_shape[1], 2 * padding)
    new_batch = []
    newPad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = numpy.lib.pad(batch[i], pad_width=newPad,
                                         mode='constant', constant_values=0)
            new_height = random.randint(0, img_shape[0] - crop_shape[0])
            new_wight = random.randint(0, img_shape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][new_height:new_height + crop_shape[0],
                           new_wight:new_wight + crop_shape[1]]
    return new_batch


# 图片数组左右移动
def random_flip_left_right(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits):
            batch[i] = numpy.fliplr(batch[i])
    return batch


# RGB数组预处理
def color_pre_process(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - numpy.mean(x_train[:, :, :, 0])) / numpy.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - numpy.mean(x_train[:, :, :, 1])) / numpy.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - numpy.mean(x_train[:, :, :, 2])) / numpy.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - numpy.mean(x_test[:, :, :, 0])) / numpy.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - numpy.mean(x_test[:, :, :, 1])) / numpy.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - numpy.mean(x_test[:, :, :, 2])) / numpy.std(x_test[:, :, :, 2])

    return x_train, x_test


# 返回新的图片数组batch
def data_augmentation(batch):
    batch = random_flip_left_right(batch)
    batch = random_crop(batch, [32, 32], 4)
    return batch

# fetch_data()
