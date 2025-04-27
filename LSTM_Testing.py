""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
from keras.utils import to_categorical


def encoding(note_idx, duration_idx, n_vocab, n_duration):
    if (n_vocab >= n_duration):
        return n_vocab * duration_idx + note_idx
    else:
        return n_duration * note_idx + duration_idx

def decoding(num, n_vocab, n_duration):
    if (n_vocab >= n_duration):
        return num % n_vocab, num // n_vocab
    else:
        return num // n_duration, num % n_duration
def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('notes_chopin', 'rb') as filepath:
        notes = pickle.load(filepath)
    with open('note_durations_chopin', 'rb') as filepath:
        note_durations = pickle.load(filepath)

    pitchnames = sorted(set(item for item in notes))
    durations_set = sorted(set(item for item in note_durations))

    # Get all pitch names
    n_vocab = len(set(notes))
    n_durations = len(set(note_durations))

    combined_input, total_output = prepare_sequences(notes, note_durations, n_vocab, n_durations)
    model = create_network(n_vocab, n_durations)
    prediction_output = generate_notes(model, combined_input, pitchnames, durations_set, n_vocab, n_durations)
    create_midi(prediction_output)

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

    network_input = network_input #/ float(n_vocab)
    duration_input = duration_input #/ float(n_durations)

    combined_input = numpy.concatenate((network_input, duration_input), axis=1)

    total_output = []
    for i in range(0, len(note_output)):
        total_output.append(encoding(note_output[i], duration_output[i], n_vocab, n_durations))

    # convert output to categorical
    #final_output = to_categorical(total_output, num_classes=n_vocab * n_durations)

    return combined_input, total_output


def create_network(n_vocab, n_durations):
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

    # Load the weights to each node
    model.load_weights('weights-improvement-150-1.7389-bigger.keras')

    return model

def generate_notes(model, network_input, pitchnames, durations_set, n_vocab, n_durations):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)
    print(start)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    int_to_duration = dict((number, duration) for number, duration in enumerate(durations_set))
    decoded_dict = {}
    for i in range(0, n_vocab * n_durations):
        x, y = decoding(i, n_vocab, n_durations)
        decoded_dict[i] = (int_to_note[x], int_to_duration[y])

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab * n_durations)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = decoded_dict[index]
        prediction_output.append(result)

        pattern = numpy.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output
def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """

    note_list = []
    duration_list = []
    for i in range(0, len(prediction_output)):
        note_list.append(prediction_output[i][0])
        duration_list.append(prediction_output[i][1])

    offset = 0
    output_notes = []
    output_durations = []

    # create note and chord objects based on the values generated by the model
    for i, pattern in enumerate(note_list):
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note), quaterLength = duration_list[i])
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes, quarterLength = duration_list[i])
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.3

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate()
