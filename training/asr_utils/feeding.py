import pandas as pd
from functools import partial
from .config import Config
from .text import text_to_char_array
from .spectrogram_augumentation import *
import numpy as np
from .flags import FLAGS
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio

def to_sparse_tuple(sequence):
    r"""Creates a sparse representention of ``sequence``.
        Returns a tuple with (indices, values, shape)
    """
    indices = np.asarray(list(zip([0]*len(sequence), range(len(sequence)))), dtype=np.int64)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)
    return indices, sequence, shape


def samples_to_mfccs(samples, sample_rate, train_phase=False):
    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=Config.audio_window_samples,
                                                  stride=Config.audio_step_samples,
                                                  magnitude_squared=True)

    # Data Augmentations
    if train_phase:
        if FLAGS.augmentation_spec_dropout_keeprate < 1:
            spectrogram = augment_dropout(spectrogram,
                                          keep_prob=FLAGS.augmentation_spec_dropout_keeprate)

        if FLAGS.augmentation_freq_and_time_masking:
            spectrogram = augment_freq_time_mask(spectrogram,
                                                 frequency_masking_para=FLAGS.augmentation_freq_and_time_masking_freq_mask_range,
                                                 time_masking_para=FLAGS.augmentation_freq_and_time_masking_time_mask_range,
                                                 frequency_mask_num=FLAGS.augmentation_freq_and_time_masking_number_freq_masks,
                                                 time_mask_num=FLAGS.augmentation_freq_and_time_masking_number_time_masks)

        if FLAGS.augmentation_pitch_and_tempo_scaling:
            spectrogram = augment_pitch_and_tempo(spectrogram,
                                                  max_tempo=FLAGS.augmentation_pitch_and_tempo_scaling_max_tempo,
                                                  max_pitch=FLAGS.augmentation_pitch_and_tempo_scaling_max_pitch,
                                                  min_pitch=FLAGS.augmentation_pitch_and_tempo_scaling_min_pitch)

        if FLAGS.augmentation_speed_up_std > 0:
            spectrogram = augment_speed_up(spectrogram, speed_std=FLAGS.augmentation_speed_up_std)

    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])

    return mfccs, tf.shape(input=mfccs)[0]


def audiofile_to_features(wav_filename, train_phase=False):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate, train_phase=train_phase)

    if train_phase:
        if FLAGS.data_aug_features_multiplicative > 0:
            features = features*tf.random.normal(mean=1, stddev=FLAGS.data_aug_features_multiplicative, shape=tf.shape(features))

        if FLAGS.data_aug_features_additive > 0:
            features = features+tf.random.normal(mean=0.0, stddev=FLAGS.data_aug_features_additive, shape=tf.shape(features))

    return features, features_len

def entry_to_features(wav_filename, transcript, train_phase):
    features, features_len = audiofile_to_features(wav_filename, train_phase=train_phase)
    return wav_filename, features, features_len, tf.SparseTensor(*transcript)


def create_dataset(csvs, batch_size, cache_path='', train_phase=False):
    df = pd.read_csv(csvs)
    df.sort_values(by="wav_filesize", inplace=True)

    try:
        df = df.apply(partial(text_to_char_array, alplabets=Config.alphabet),
                      result_type='boardcast', axis=1)
    except ValueError as e:
        err_msg, series, _ = e.args
        print(err_msg)
        exit(1)

    def generate_values():
        for i, row in df.iterrows():
            yield row.wav_filename, to_sparse_tuple(row.transcript)

    def sparse_reshape(sparse):
        shape = sparse.dense_shape
        return tf.sparse.reshape(sparse, [shape[0], shape[2]])

    def batch_fn(wav_filenames, features, features_len, transcripts):
        features = tf.data.Dataset.zip((features, features_len))
        features =  features.padded_batch(batch_size,
                                          padded_shapes=([None, Config.n_input, []]))
        transcripts = transcripts.batch(batch_size).map(sparse_reshape)
        wav_filenames = wav_filenames.batch(batch_size)
        return tf.data.Dataset.zip((wav_filenames, features, transcripts))

    process_fn = partial(entry_to_features, train_phase=train_phase)

    dataset = (tf.data.Dataset.from_generator(generate_values,
                                              output_types=(tf.string, (tf.int64, tf.int32, tf.int64)))
                        .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE))

    dataset = (dataset.window(batch_size, drop_remainder=True).flat_map(batch_fn).prefetch(1))

    return dataset