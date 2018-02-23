import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_minibatch(dataset, ratio_tr=None, ratio_val=None, binarise=False, path_datadir='../datasets',
                   size_minibatch=128, size_testbatch=-1, nb_towers=1, nb_threads=2, seed_split=0, seed_minibatch=0,
                   dtype=tf.float32, name='data_prep', noise_level=0.1):
    with tf.name_scope(name):
        if dataset in ['mnist', 'mnist-small', 'fashion']:
            if ratio_tr is not None:
                print("Warning: ratio_tr is predefined for the '%s' dataset." % dataset)

            # data = mnist.input_data.read_data_sets(path_datadir + '/' + dataset, one_hot=False)
            filename_queue_tr = tf.train.string_input_producer(
                [path_datadir + '/' + dataset + '_new/train.tfrecords'])
            X_tr, y_tr = read_from_tfrec_file(filename_queue_tr, 784, binarise=binarise)

            if ratio_val is None:
                filename_queue_te = tf.train.string_input_producer(
                    [path_datadir + '/' + dataset + '_new/test.tfrecords'])
            else:
                filename_queue_te = tf.train.string_input_producer(
                    [path_datadir + '/' + dataset + '_new/validation.tfrecords'])
            X_te, y_te = read_from_tfrec_file(filename_queue_te, 784, binarise=binarise)

            # transform labels to one-hot-vectors
            nb_classes = 10
            y_tr = tf.one_hot(y_tr, nb_classes)
            y_te = tf.one_hot(y_te, nb_classes)

        else:
            if binarise:
                raise NotImplementedError

            # load data
            labels = None
            if dataset == 'geyser':
                path = path_datadir + '/geyser'
                data = pd.read_csv(path, sep=' ', header=None).values[:, [1, 2]]
                labels = (data[:, 1] > 20).astype(np.int)

            elif dataset == 'pinwheel' or dataset == 'noisy-pinwheel':
                nb_spokes = 5
                data, labels = make_pinwheel_data(0.3, 0.05, nb_spokes, 200, 0.25)

            elif dataset == 'aggregation':
                path = path_datadir + '/Aggregation.txt'
                data_lbo = pd.read_csv(path, sep='\t', header=None).values
                data = data_lbo[:, 0:2]
                labels = data_lbo[:, 2].astype(np.int) - 1

            elif dataset == 'auto':
                path = path_datadir + '/Auto/auto-mpg.csv'
                data = pd.read_csv(path, sep=',', header=None).values
                data = data[data[:, 3] != '?']
                labels = data[:, 1]  # nb cylinders
                # labels = data[:, 7]  # origin
                feats = [0, 2, 3, 4, 5, 6]  # mpg, displacement, horsepower, weight, acceleration, model_year (discr)
                data = data[:, feats].astype(np.float64)
                labels[labels == 3] = 0
                labels[labels == 4] = 1
                labels[labels == 5] = 2
                labels[labels == 6] = 3
                labels[labels == 8] = 4
                # labels = labels - 1
                labels = labels.astype(np.int)

            else:
                raise Exception("Dataset '%s' does not exist." % dataset)

            N = data.shape[0]

            # transform labels to 1-hot vectors
            if labels is not None:
                labels1h = np.zeros((N, np.unique(labels).size))
                labels1h[range(N), labels] = 1

            # make tr-val-te split...
            ratio_te = 1 - ratio_tr
            if labels is None:
                X_tr, X_te = train_test_split(data, test_size=ratio_te, random_state=seed_split)
                y_tr, y_te = None, None
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(data, labels1h, test_size=ratio_te, random_state=seed_split)

            N_te = X_te.shape[0]
            if ratio_val is not None:
                # compute validation ratio w.r.t training set
                assert ratio_tr > ratio_val
                ratio_te -= ratio_val
                ratio_val_tr = ratio_val / (ratio_tr + ratio_val)

                # split validation set
                if labels is None:
                    X_tr, X_te = train_test_split(X_tr, test_size=ratio_val_tr, random_state=seed_split)
                else:
                    X_tr, X_te, y_tr, y_te = train_test_split(X_tr, y_tr, test_size=ratio_val_tr,
                                                              random_state=seed_split)
                print('N = %6d\n\tN_tr =  %6d\n\tN_val = %6d\n\tN_te =  %6d\n' % (N, X_tr.shape[0],
                                                                                  X_te.shape[0], N_te))
            else:
                print(' N = %6d\n\tN_tr = %6d\n\tN_te = %6d\n' % (N, X_tr.shape[0], N_te))

            # perturb training data if necessary
            if dataset == 'noisy-pinwheel':
                X_tr = perturb_data(X_tr, noise_ratio=noise_level, noise_mean=0, noise_stddev=10, seed=seed_split)

            # apply scaling if necessary
            if dataset == 'auto':
                scaler = StandardScaler().fit(X_tr)
                X_tr = scaler.transform(X_tr) * 5
                X_te = scaler.transform(X_te) * 5
            elif dataset not in ['pinwheel', 'noisy-pinwheel']:
                scaler = StandardScaler().fit(X_tr)
                X_tr = scaler.transform(X_tr)
                X_te = scaler.transform(X_te)

            # create tensors
            X_tr = tf.constant(X_tr, name='train_set', dtype=dtype)
            X_te = tf.constant(X_te, name='test_set', dtype=dtype)
            if labels is not None:
                y_tr = tf.constant(y_tr, name='train_labels', dtype=dtype)
                y_te = tf.constant(y_te, name='test_labels', dtype=dtype)

        # setup train minibatching (if size_minibatch is negative, return full dataset)
        if size_minibatch > 0:
            # make sure that there are enough samples in the queue to have enough randomness
            min_after_dequeue = 100 * size_minibatch

            # compute queue capacity as suggested here: https://www.tensorflow.org/api_guides/python/reading_data
            capacity = min_after_dequeue + (nb_threads + 1) * size_minibatch

            # make shuffle queues
            enq_many = dataset not in ['mnist', 'mnist-small', 'fashion']
            if y_tr is None:
                m_batch = tf.train.shuffle_batch([X_tr], batch_size=size_minibatch, capacity=capacity,
                                                 enqueue_many=enq_many, min_after_dequeue=min_after_dequeue,
                                                 num_threads=nb_threads, seed=seed_minibatch)
                m_batch_lbl = None
            else:
                m_batch, m_batch_lbl = tf.train.shuffle_batch([X_tr, y_tr], batch_size=size_minibatch,
                                                              capacity=capacity, enqueue_many=enq_many,
                                                              min_after_dequeue=min_after_dequeue,
                                                              num_threads=nb_threads, seed=seed_minibatch)
        else:
            m_batch = X_tr
            m_batch_lbl = y_tr

        # setup test minibatching
        if size_testbatch > 0:
            min_after_dequeue = 100 * size_testbatch
            capacity = min_after_dequeue + (nb_threads + 1) * size_testbatch

            enq_many = dataset not in ['mnist', 'mnist-small', 'fashion']
            if y_te is None:
                m_batch_te = tf.train.shuffle_batch([X_te], batch_size=size_testbatch, capacity=capacity,
                                                    enqueue_many=enq_many, min_after_dequeue=min_after_dequeue,
                                                    num_threads=nb_threads, seed=seed_minibatch)
                m_batch_te_lbl = None
            else:
                m_batch_te, m_batch_te_lbl = tf.train.shuffle_batch([X_te, y_te], batch_size=size_testbatch,
                                                                    capacity=capacity, enqueue_many=enq_many,
                                                                    min_after_dequeue=min_after_dequeue,
                                                                    num_threads=nb_threads, seed=seed_minibatch)
        else:
            m_batch_te = X_te
            m_batch_te_lbl = y_te

        # split minibatch to share it across towers
        if nb_towers > 1:
            m_batch = tf.split(m_batch, nb_towers, axis=0, num=nb_towers, name='data_split')
        return m_batch, m_batch_lbl, m_batch_te, m_batch_te_lbl


def read_from_tfrec_file(filename_q, D, binarise=False, seed=0):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_q)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has length mnist.IMAGE_PIXELS) to a uint8 tensor with
    # shape [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([D])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255)

    # Convert from [0, 1] to {-1, 1} if required
    if binarise:
        # stochastically binarise: pixel intensities are used as probabilities
        image = tf.reshape(image, (-1, 1))
        probs = tf.concat([1. - image, image], axis=1)
        samps = tf.reshape(tf.multinomial(logits=tf.log(probs), num_samples=1, seed=seed), (D,))
        image = tf.where(tf.equal(samps, 0),
                         -tf.ones_like(samps, dtype=tf.float32),
                         tf.ones_like(samps, dtype=tf.float32))
        # thresholding
        # image = tf.where(image < 0.5, -tf.ones_like(image, dtype=tf.float32), tf.ones_like(image, dtype=tf.float32))

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    # code from Johnson et. al. (2016)
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    np.random.seed(1)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum('ti,tij->tj', features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))

    return data[:, 0:2], data[:, 2].astype(np.int)


def perturb_data(x, noise_ratio=0.1, noise_mean=0, noise_stddev=10, seed=0):
    """
    Replace random points by random noise
    Args:
        x: dataset
        noise_ratio: ratio of datapoints to perturb
        loc: noise mean
        scale: noise stddev
    Returns:
        perturbed dataset
    """
    np.random.seed(seed)

    # choose datapoints to perturb
    N, D = x.shape
    N_noise = int(N * noise_ratio)
    noise_indices = np.random.permutation(np.arange(N))[:N_noise]

    # perturb data: add noise (normal distributed with mean=0 and stddev=10)
    x[noise_indices, :] = np.random.normal(loc=noise_mean, scale=noise_stddev, size=(N_noise, D))

    return x
