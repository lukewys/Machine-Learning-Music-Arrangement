import numpy as np
from tensorflow.contrib import rnn
from matplotlib import pyplot as plt
import datetime
import random
import pretty_midi
import ast
import os
import time
import tensorflow as tf
import magenta
from magenta.models.polyphony_rnn import polyphony_model
from magenta.models.polyphony_rnn import polyphony_sequence_generator
from magenta.music import constants
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

abspath = os.path.abspath('.')
path = abspath + '\\results'
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)
path = abspath + '\\results\\mono'
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)

global batch_count
batch_count = 0
global epoch_count
epoch_count = 0
RANDOM_OUT = True
HIDDEN_SIZE = 128
NUM_LAYERS = 2
TIMESTEPS = 256
learning_rate = 0.00001
ONEHOT_SIZE = 128
RECUR_STEP = 256
sess = tf.InteractiveSession()


def clear_count():
    global batch_count
    batch_count = 0
    global epoch_count
    epoch_count = 0


def elapsed(sec):
    if sec < 60:
        return "{:.2f}".format(sec) + " sec"
    elif sec < (60 * 60):
        return "{:.2f}".format(sec / 60) + " min"
    else:
        return "{:.2f}".format(sec / (60 * 60)) + " hr"


def generate_data(seq, TIMESTEPS):
    X = []
    Y = []
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def get_data(num, data, TIMESTEPS):
    if num == -1:
        batch_x, batch_y = generate_data(data, TIMESTEPS)
        batch_x = batch_x.reshape(-1, TIMESTEPS, 1)
        return batch_x, batch_y
    n = len(data)
    r = np.random.random_integers(0, n - num - TIMESTEPS - 2)
    seq = data[r:r + num + TIMESTEPS + 1]
    batch_x, batch_y = generate_data(seq, TIMESTEPS)
    batch_x = batch_x.reshape(-1, TIMESTEPS, 1)
    return batch_x, batch_y


def get_data_n(num, data, TIMESTEPS):
    if num == -1:
        batch_x, batch_y = generate_data(data, TIMESTEPS)
        batch_x = batch_x.reshape(-1, TIMESTEPS, 1)
        return batch_x, batch_y
    global batch_count
    global epoch_count
    n = len(data)
    r = batch_count % (n - num - TIMESTEPS - 2)
    seq = data[r:r + num + TIMESTEPS + 1]
    batch_x, batch_y = generate_data(seq, TIMESTEPS)
    batch_x = batch_x.reshape(-1, TIMESTEPS, 1)
    FIRST = batch_count == 0 and epoch_count == 0
    batch_count += num
    if r < num and not FIRST:
        epoch_count += 1
        print('EPOCH ' + str(epoch_count))
    return batch_x, batch_y


def make_batch(num, data, ONEHOT_SIZE, TIMESTEPS):
    batch_xs, batch_ys = get_data(num, data, TIMESTEPS)
    batch_ys = onehot_gen(batch_ys, ONEHOT_SIZE).reshape((-1, ONEHOT_SIZE))
    return batch_xs, batch_ys


def make_batch_onehot(num, data, ONEHOT_SIZE, TIMESTEPS):
    batch_xs, batch_ys = get_data(num, data, TIMESTEPS)
    batch_xs = onehot_gen(batch_xs, ONEHOT_SIZE).reshape((-1, TIMESTEPS, ONEHOT_SIZE))
    batch_ys = onehot_gen(batch_ys, ONEHOT_SIZE).reshape((-1, ONEHOT_SIZE))
    return batch_xs, batch_ys


def make_batch_n(num, data, ONEHOT_SIZE, TIMESTEPS):
    batch_xs, batch_ys = get_data_n(num, data, TIMESTEPS)
    batch_ys = onehot_gen(batch_ys, ONEHOT_SIZE).reshape((-1, ONEHOT_SIZE))
    return batch_xs, batch_ys


def make_batch_n_onehot(num, data, ONEHOT_SIZE, TIMESTEPS):
    batch_xs, batch_ys = get_data_n(num, data, TIMESTEPS)
    batch_xs = onehot_gen(batch_xs, ONEHOT_SIZE).reshape((-1, TIMESTEPS, ONEHOT_SIZE))
    batch_ys = onehot_gen(batch_ys, ONEHOT_SIZE).reshape((-1, ONEHOT_SIZE))
    return batch_xs, batch_ys


def onehot_gen(data, ONEHOT_SIZE):
    data = data.reshape(-1)
    m = ONEHOT_SIZE
    n = data.shape[0]
    p = np.zeros([n, m], dtype=np.float32)
    for i in range(n):
        l = int(data[i])
        p[i, l] = 1
    return p


def random_index(rate):
    start = 0
    randnum = random.uniform(0, np.sum(rate))
    for index in range(len(rate)):
        start += rate[index]
        if randnum <= start:
            break
    return index


def LstmCell(HIDDEN_SIZE):
    # lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE)
    lstm_cell = rnn.GRUCell(HIDDEN_SIZE)
    return lstm_cell


def RNN(x, weights, biases, HIDDEN_SIZE, TIMESTEPS, NUM_LAYERS):
    x = tf.reshape(x, [-1, TIMESTEPS])
    x = tf.split(x, TIMESTEPS, 1)
    rnn_cell = rnn.MultiRNNCell([LstmCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def RNN_onehot_in(x, weights, biases, HIDDEN_SIZE, ONEHOT_SIZE, TIMESTEPS, NUM_LAYERS):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, ONEHOT_SIZE])
    x = tf.matmul(x, weights['in']) + biases['in']
    x = tf.split(x, TIMESTEPS, 0)
    rnn_cell = rnn.MultiRNNCell([LstmCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def save_acc_loss(ACC, LOSS):
    now_time = datetime.datetime.now()
    str_time = datetime.datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    np.save("results/loss_" + str_time + ".npy", LOSS)
    np.save("results/acc_" + str_time + ".npy", ACC)
    plt.figure('loss')
    plt.plot(LOSS)
    plt.savefig("results/loss_" + str_time + ".png")
    plt.figure('accuracy')
    plt.plot(ACC)
    plt.savefig("results/accuracy_" + str_time + ".png")


class note:

    def __init__(self, key, duration):
        self.key = key
        self.duration = duration

    def show(self):
        print('key:' + str(self.key))
        print("duration:" + str(self.duration))


def data_to_note(data):
    notelist = []
    n = len(data)
    offset = 0
    data[-1] = 0
    while offset < n - 1:
        if data[offset] != 0:
            num = 0
            key = data[offset]
            while data[offset] == key:
                num += 1
                offset += 1
                if data[offset] == key and data[offset + 1] != key:
                    p = note(key, num)
                    notelist.append(p)
        else:
            offset += 1
    return notelist


def to_midi(velocity, rate, name, data):
    data = data.astype(int)
    l = data_to_note(data)
    offset = 0
    midi = pretty_midi.PrettyMIDI()
    midi_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    midi_instrument = pretty_midi.Instrument(program=midi_program)
    for _ in range(len(l)):
        note = (pretty_midi.Note(velocity, l[_].key, offset, offset + l[_].duration * rate))
        offset += l[_].duration * rate
        midi_instrument.notes.append(note)
    midi.instruments.append(midi_instrument)
    midi.write(name)


x = tf.placeholder("float", [None, TIMESTEPS, ONEHOT_SIZE])
y = tf.placeholder("float", [None, ONEHOT_SIZE])

weights = {
    'out': tf.Variable(tf.random_normal([HIDDEN_SIZE, ONEHOT_SIZE])),
    'in': tf.Variable(tf.random_normal([ONEHOT_SIZE, HIDDEN_SIZE]))
}
biases = {
    'out': tf.Variable(tf.random_normal([ONEHOT_SIZE])),
    'in': tf.Variable(tf.random_normal([HIDDEN_SIZE]))
}

pred = RNN_onehot_in(x, weights, biases, HIDDEN_SIZE, ONEHOT_SIZE, TIMESTEPS, NUM_LAYERS)
softmax_pred = tf.nn.softmax(pred)
# softmax_pred = tf.exp(pred) / tf.reduce_sum(tf.exp(pred))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)


def recur(N, data):
    if len(data) > TIMESTEPS:
        data = data[:TIMESTEPS]
    if len(data) < TIMESTEPS:
        l = len(data)
        z = np.zeros((TIMESTEPS - l))
        data = np.append(data, z)
    predict = np.array([], dtype=float)
    predict = np.append(predict, data.reshape(-1))
    recur_time = time.time()
    model_file = tf.train.latest_checkpoint('save/')
    saver.restore(sess, model_file)
    i = 0
    while i < N:
        xs = onehot_gen(data, ONEHOT_SIZE).reshape((-1, TIMESTEPS, ONEHOT_SIZE))
        if RANDOM_OUT == True:
            pred_ = sess.run(softmax_pred, feed_dict={x: xs}).reshape(-1)
            pred_ = random_index(pred_)
        else:
            pred_ = sess.run(pred, feed_dict={x: xs})
            pred_ = pred_.argmax(axis=1)
        predict = np.append(predict, pred_)
        data = np.append(data, pred_)
        data = data[1:]
        i += 1
    return predict


def generate_mono(filepath, times):
    FS = 32
    p = np.array([], dtype=int)
    midi_data = pretty_midi.PrettyMIDI(filepath)
    a = midi_data.instruments[0].get_piano_roll(fs=FS)
    datrc = np.append(p, np.argmax(a, axis=0))  # 取高音
    velocity = 80
    rate = 0.015625 * 2
    print("\n\n\n开始生成...\n\n\n")
    for _ in range(times):
        predicted = recur(RECUR_STEP, datrc)
        now_time = datetime.datetime.now()
        str_time = datetime.datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
        abspath = os.path.abspath('.')
        outdir = abspath + '\\results\\mono\\'
        to_midi(velocity, rate, outdir+str_time+'.mid', predicted)


##############
# magenta的模型#
##############


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'run_dir', r'',
    'Path to the directory where the latest checkpoint will be loaded from.')

tf.app.flags.DEFINE_string(
    'bundle_file',
    '',
    'Path to the bundle file. If specified, this will take priority over '
    'run_dir, unless save_generator_bundle is True, in which case both this '
    'flag and run_dir are required')
tf.app.flags.DEFINE_boolean(
    'save_generator_bundle', False,
    'If true, instead of generating a sequence, will save this generator as a '
    'bundle file in the location specified by the bundle_file flag')
tf.app.flags.DEFINE_string(
    'bundle_description', None,
    'A short, human-readable text description of the bundle (e.g., training '
    'data, hyper parameters, etc.).')
tf.app.flags.DEFINE_string(
    'config', 'polyphony', 'Config to use.')
tf.app.flags.DEFINE_string(
    'output_dir',
    '',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 10,
    'The number of tracks to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_integer(
    'num_steps', 512,
    'The total number of steps the generated track should be, priming '
    'track length + generated steps. Each step is a 16th of a bar.')
tf.app.flags.DEFINE_string(
    'primer_pitches', '',
    'A string representation of a Python list of pitches that will be used as '
    'a starting chord with a quarter note duration. For example: '
    '"[60, 64, 67]"')
tf.app.flags.DEFINE_string(
    'primer_melody', '',
    'A string representation of a Python list of '
    'magenta.music.Melody event values. For example: '
    '"[60, -2, 60, -2, 67, -2, 67, -2]".')
tf.app.flags.DEFINE_string(
    'primer_midi', r'',
    'The path to a MIDI file containing a polyphonic track that will be used '
    'as a priming track.')
tf.app.flags.DEFINE_boolean(
    'condition_on_primer', False,
    'If set, the RNN will receive the primer as its input before it begins '
    'generating a new sequence.')
tf.app.flags.DEFINE_boolean(
    'inject_primer_during_generation', True,
    'If set, the primer will be injected as a part of the generated sequence. '
    'This option is useful if you want the model to harmonize an existing '
    'melody.')
tf.app.flags.DEFINE_float(
    'qpm', None,
    'The quarters per minute to play generated output at. If a primer MIDI is '
    'given, the qpm from that will override this flag. If qpm is None, qpm '
    'will default to 120.')
tf.app.flags.DEFINE_float(
    'temperature', 1.0,
    'The randomness of the generated tracks. 1.0 uses the unaltered '
    'softmax probabilities, greater than 1.0 makes tracks more random, less '
    'than 1.0 makes tracks less random.')
tf.app.flags.DEFINE_integer(
    'beam_size', 1,
    'The beam size to use for beam search when generating tracks.')
tf.app.flags.DEFINE_integer(
    'branch_factor', 1,
    'The branch factor to use for beam search when generating tracks.')
tf.app.flags.DEFINE_integer(
    'steps_per_iteration', 1,
    'The number of steps to take per beam search iteration.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')
tf.app.flags.DEFINE_string(
    'hparams', 'batch_size=16,rnn_layer_sizes=[64,64]',
    'Comma-separated list of `name=value` pairs. For each pair, the value of '
    'the hyperparameter named `name` is set to `value`. This mapping is merged '
    'with the default hyperparameters.')


def get_checkpoint():
    """Get the training dir or checkpoint path to be used by the model."""
    if FLAGS.run_dir and FLAGS.bundle_file and not FLAGS.save_generator_bundle:
        raise magenta.music.SequenceGeneratorException(
            'Cannot specify both bundle_file and run_dir')
    if FLAGS.run_dir:
        train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')
        return train_dir
    else:
        return None


def get_bundle():
    """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.

    Returns:
      Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
      not set or the save_generator_bundle flag is set.
    """
    if FLAGS.save_generator_bundle:
        return None
    if FLAGS.bundle_file is None:
        return None
    bundle_file = os.path.expanduser(FLAGS.bundle_file)
    return magenta.music.read_bundle_file(bundle_file)


def run_with_flags(generator):
    """Generates polyphonic tracks and saves them as MIDI files.

    Uses the options specified by the flags defined in this module.

    Args:
      generator: The PolyphonyRnnSequenceGenerator to use for generation.
    """
    if not FLAGS.output_dir:
        tf.logging.fatal('--output_dir required')
        return
    output_dir = os.path.expanduser(FLAGS.output_dir)

    primer_midi = None
    if FLAGS.primer_midi:
        primer_midi = os.path.expanduser(FLAGS.primer_midi)

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    primer_sequence = None
    qpm = FLAGS.qpm if FLAGS.qpm else magenta.music.DEFAULT_QUARTERS_PER_MINUTE
    if FLAGS.primer_pitches:
        primer_sequence = music_pb2.NoteSequence()
        primer_sequence.tempos.add().qpm = qpm
        primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ
        for pitch in ast.literal_eval(FLAGS.primer_pitches):
            note = primer_sequence.notes.add()
            note.start_time = 0
            note.end_time = 60.0 / qpm
            note.pitch = pitch
            note.velocity = 100
        primer_sequence.total_time = primer_sequence.notes[-1].end_time
    elif FLAGS.primer_melody:
        primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
        primer_sequence = primer_melody.to_sequence(qpm=qpm)
    elif primer_midi:
        primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
        if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
            qpm = primer_sequence.tempos[0].qpm
    else:
        tf.logging.warning(
            'No priming sequence specified. Defaulting to empty sequence.')
        primer_sequence = music_pb2.NoteSequence()
        primer_sequence.tempos.add().qpm = qpm
        primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ

    # Derive the total number of seconds to generate.
    seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
    generate_end_time = FLAGS.num_steps * seconds_per_step

    # Specify start/stop time for generation based on starting generation at the
    # end of the priming sequence and continuing until the sequence is num_steps
    # long.
    generator_options = generator_pb2.GeneratorOptions()
    # Set the start time to begin when the last note ends.
    generate_section = generator_options.generate_sections.add(
        start_time=primer_sequence.total_time,
        end_time=generate_end_time)

    if generate_section.start_time >= generate_section.end_time:
        tf.logging.fatal(
            'Priming sequence is longer than the total number of steps '
            'requested: Priming sequence length: %s, Total length '
            'requested: %s',
            generate_section.start_time, generate_end_time)
        return

    generator_options.args['temperature'].float_value = FLAGS.temperature
    generator_options.args['beam_size'].int_value = FLAGS.beam_size
    generator_options.args['branch_factor'].int_value = FLAGS.branch_factor
    generator_options.args[
        'steps_per_iteration'].int_value = FLAGS.steps_per_iteration

    generator_options.args['condition_on_primer'].bool_value = (
        FLAGS.condition_on_primer)
    generator_options.args['no_inject_primer_during_generation'].bool_value = (
        not FLAGS.inject_primer_during_generation)

    tf.logging.debug('primer_sequence: %s', primer_sequence)
    tf.logging.debug('generator_options: %s', generator_options)

    # Make the generate request num_outputs times and save the output as midi
    # files.
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    digits = len(str(FLAGS.num_outputs))
    for i in range(FLAGS.num_outputs):
        generated_sequence = generator.generate(primer_sequence, generator_options)

        midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
        midi_path = os.path.join(output_dir, midi_filename)
        magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

    tf.logging.info('Wrote %d MIDI files to %s',
                    FLAGS.num_outputs, output_dir)


def main(unused_argv):
    """Saves bundle or runs generator based on flags."""
    tf.logging.set_verbosity(FLAGS.log)

    bundle = get_bundle()

    config_id = bundle.generator_details.id if bundle else FLAGS.config
    config = polyphony_model.default_configs[config_id]
    config.hparams.parse(FLAGS.hparams)
    # Having too large of a batch size will slow generation down unnecessarily.
    config.hparams.batch_size = min(
        config.hparams.batch_size, FLAGS.beam_size * FLAGS.branch_factor)

    generator = polyphony_sequence_generator.PolyphonyRnnSequenceGenerator(
        model=polyphony_model.PolyphonyRnnModel(config),
        details=config.details,
        steps_per_quarter=config.steps_per_quarter,
        checkpoint=get_checkpoint(),
        bundle=bundle)

    if FLAGS.save_generator_bundle:
        bundle_filename = os.path.expanduser(FLAGS.bundle_file)
        if FLAGS.bundle_description is None:
            tf.logging.warning('No bundle description provided.')
        tf.logging.info('Saving generator bundle to %s', bundle_filename)
        generator.create_bundle_file(bundle_filename, FLAGS.bundle_description)
    else:
        run_with_flags(generator)


def generate_poly(filepath, mode):
    FLAGS.primer_midi = filepath
    abspath = os.path.abspath('.')
    FLAGS.output_dir = abspath + '\\results\\poly'
    if mode == 'bach':
        FLAGS.bundle_file = abspath + '\\poly_bach.mag'
    if mode == 'mix':
        FLAGS.bundle_file = abspath + '\\poly_mix.mag'
    print("\n\n\n开始生成...\n\n\n")
    tf.app.run(main)
