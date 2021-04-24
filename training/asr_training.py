from .asr_utils.flags import create_flags, FLAGS
from .asr_utils.feeding import create_dataset
import absl
import tensorflow as tf


def create_optimizer():
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                                 beta1=FLAGS.beta1,
                                                 beta2=FLAGS.beta2,
                                                 epsilon=FLAGS.epsilon)
    return optimizer


def train():
    do_cache_dataset = True

    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=FLAGS.train_batch_size,
                               cache_path=FLAGS.feature_cache if do_cache_dataset else None)

    iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_set),
                                                         tf.compat.v1.data.get_output_shapes(train_set),
                                                         output_classes=tf.compat.v1.data.get_output_classes(train_set))
    train_init_op = iterator.make_initializer(train_set)

    optimizer = create_optimizer()



def main():
    if FLAGS.train_files:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(FLAGS.random_seed)
        train()


if __name__ == '__main__':
    create_flags()
    main()