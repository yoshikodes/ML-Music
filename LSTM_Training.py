import glob
import pickle
import numpy
import pretty_midi
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def train_network():
    """ Train a Neural Network to generate music """
    notes, note_durations = get_notes()

    # get amount of unique pitches and durations
    n_vocab = len(set(notes))
    n_durations = len(set(note_durations))

    network_input, duration_input, final_output = prepare_sequences(
        notes, note_durations, n_vocab, n_durations)

    model = create_network(network_input, n_vocab, n_durations)

    train(model, network_input, duration_input, final_output)

def get_notes():
    """ Get all the notes and chords from the midi files """
    notes = []
    note_durations = []

    for file in glob.glob("archive/chopin/*.mid"):
        midi = pretty_midi.PrettyMIDI(file)
        for instrument in midi.instruments:
            instrument.control_changes = [cc for cc in instrument.control_changes if cc.number != 64]
        midi.write(file)
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                note_durations.append(float(element.duration.quarterLength))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                note_durations.append(float(element.duration.quarterLength))

    with open('notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    with open('note_durations', 'wb') as filepath:
        pickle.dump(note_durations, filepath)

    return notes, note_durations

def encoding(note_idx, duration_idx, n_vocab, n_duration):
    if (n_vocab >= n_duration):
        return n_vocab * duration_idx + note_idx
    else:
        return n_duration * note_idx + duration_idx

def decoding(num, n_vocab, n_duration):
    if (n_vocab >= n_duration):
        return num // n_vocab, num % n_vocab
    else:
        return num // n_duration, num % n_duration

def prepare_sequences(notes, durations, n_vocab, n_durations):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 50  # Reduced from 100 since we're handling both notes and durations

    # get all pitch names and durations
    pitchnames = sorted(set(item for item in notes))
    durations_set = sorted(set(item for item in durations))

    # create a dictionary to map pitches and durations to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    duration_to_int = dict((duration, number) for number, duration in enumerate(durations_set))

    network_input = []
    duration_input = []
    note_output = []
    duration_output = []

    # create input sequences and corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        duration_in = durations[i:i + sequence_length]
        note_output.append(note_to_int[notes[i + sequence_length]])
        duration_output.append(duration_to_int[durations[i + sequence_length]])
        network_input.append([note_to_int[char] for char in sequence_in])
        duration_input.append([duration_to_int[dur] for dur in duration_in])

    n_patterns = len(network_input)

    # reshape and normalize input
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    duration_input = numpy.reshape(duration_input, (n_patterns, sequence_length, 1))

    network_input = network_input / float(n_vocab)
    duration_input = duration_input / float(n_durations)

    total_output = []
    for i in range(0, len(note_output)):
        total_output.append(encoding(note_output[i], duration_output[i], n_vocab, n_durations))

    # convert output to categorical
    final_output = to_categorical(total_output, num_classes=n_vocab * n_durations)

    return network_input, duration_input, final_output

def create_network(network_input, n_vocab, n_durations):
    """ create the structure of the neural network """
    model = Sequential()

    # First LSTM layer processes note sequences
    model.add(LSTM(
        512,
        input_shape=(100, 1),
        recurrent_dropout=0.3,
        return_sequences=True
    ))

    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))

    # Dense layers to process combined features
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))

    # Output layers - one for notes and one for durations
    model.add(Dense(n_vocab * n_durations))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, duration_input, final_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # Combine inputs and outputs for training
    #print(network_input.shape, duration_input.shape, len(note_output), len(duration_output))
    combined_input = numpy.concatenate((network_input, duration_input), axis=1)
    #combined_output = numpy.concatenate((note_output, duration_output), axis=1)

    model.fit(combined_input, final_output, epochs=100, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
