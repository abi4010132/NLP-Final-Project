# import
import pandas as pd
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Activation, concatenate
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from gensim.models import Word2Vec

MAXIMUM_LENGTH = 30
DIMENSIONALITY = 128
BATCH_SIZE = 32
EPOCHS = 2
EMBEDDING_DIM = 100
w2v = None
vocab = None

def get_data(filename):
    data = pd.read_csv(filename)
    data["Question"] = data["Question"].apply(lambda x: eval(x))
    data["Question"] = data["Question"].apply(lambda x: x[:min(MAXIMUM_LENGTH, len(x))])
    data["Question"] = data["Question"].apply(lambda x: x[:len(x) - 1] + ["<eos>"])
    data["Answer"] = data["Answer"].apply(lambda x: eval(x))
    data["Answer"] = data["Answer"].apply(lambda x: x[:min(MAXIMUM_LENGTH, len(x))])
    data["Answer"] = data["Answer"].apply(lambda x: x[:len(x) - 1] + ["<eos>"])
    print("Done with loading")
    return data

def custom_loss(y_true, y_pred):
    ssc = SparseCategoricalCrossentropy()
    loss = ssc(y_true, y_pred)
    return loss

def create_embeddings(data):
    # word2vec the data
    sentences = data['Question'] + data['Answer']
    w2v = Word2Vec(sentences=sentences, min_count=1, vector_size=EMBEDDING_DIM, workers=8)
    number_of_tokens = len(w2v.wv.key_to_index)
    print("Done with W2V")
    
    vocab = w2v.wv
    embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
    for i in range(len(vocab)):
        embedding_matrix[i] = vocab[i]
    
    print("Done with embedding_matrix")
    return embedding_matrix, w2v, vocab

def create_sparse_encoding(w2v, vocab, data):
    maximum_length_input = np.max([len(d) for d in data["Question"]])
    maximum_length_output = np.max([len(d) for d in data["Answer"]])

    # i/o data
    words_input_indices = [[(w2v.wv.key_to_index[dt] if dt in vocab else 0) for dt in d] for d in data["Question"]]
    encoder_input_data = np.array([np.pad(x, (0, maximum_length_input - len(x)))for x in words_input_indices])
    words_output_indices = [[(w2v.wv.key_to_index[dt] if dt in vocab else 0) for dt in d] for d in data["Answer"]]
    decoder_input_data = np.array([np.pad(x, (0, maximum_length_output - len(x)))for x in words_output_indices])
    
    
    arr1 = decoder_input_data[:,1:]
    arr2 = decoder_input_data[:,-1:]
    decoder_output_data = np.concatenate((arr1, arr2), axis=1)
    print("Done with vectors")
    
    return encoder_input_data, decoder_input_data, decoder_output_data

def build_model(hp, data):
    
    embedding_matrix, w2v, vocab = create_embeddings(data)

    #Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(len(vocab), EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(encoder_inputs)
    encoder_lstm1 = LSTM(DIMENSIONALITY, return_sequences=True, return_state=True)
    encoder_outputs, state_hidden, state_cell = encoder_lstm1(encoder_embedding)
    encoder_states = [state_hidden, state_cell]
    #Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(len(vocab), EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(decoder_inputs)
    decoder_lstm1 = LSTM(DIMENSIONALITY, return_sequences=True, return_state=True)
    decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm1(decoder_embedding, initial_state=encoder_states)

    # Luong Attention 
    # allignment scores are dot product hidden states encoder/decoder
    allignment = np.dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(allignment)

    # context vector is dot product allignment scores and encoder hidden states
    context = np.dot([attention, encoder_outputs], axes=[2, 1])
        
    # Concatenate context vector and hidden decoder states
    context_decoder_outputs = concatenate([context, decoder_outputs])

    decoder_dense = Dense(len(vocab), activation='softmax')

    # Dense output layer of the model
    decoder_outputs = decoder_dense(context_decoder_outputs)

    #Model
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    #filepath="weights-improvement-attention-{epoch:02d}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, save_weights_only=False, mode='auto', save_freq=1)
    early_stopping = EarlyStopping(monitor='loss', patience=1)
    #callbacks_list = [checkpoint, early_stopping]
    callbacks_list = [early_stopping]

    #Compiling
    training_model.compile(optimizer='rmsprop', loss=custom_loss, metrics=['accuracy'], sample_weight_mode='temporal')

    encoder_input_data, decoder_input_data, decoder_output_data = create_sparse_encoding(w2v, vocab, data)

    #Training
    training_model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split=0.2, callbacks=callbacks_list)

    # Encoder model
    encoder_model = Model(encoder_inputs, [encoder_outputs, state_hidden, state_cell])
    encoder_states = [state_hidden, state_cell]

    # Decoder model
    decoder_state_input_hidden = Input(shape=(DIMENSIONALITY,))
    decoder_state_input_cell = Input(shape=(DIMENSIONALITY,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm1(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [decoder_state_hidden, decoder_state_cell]


    # input layer to plug encoder hidden state sequences into decoder
    decoder_context_inputs = Input(shape=(None, DIMENSIONALITY))

    # compute allignment and attention scores
    allignment = dot([decoder_outputs, decoder_context_inputs], axes=[2, 2])
    attention = Activation('softmax')(allignment)

    # context vector is dot product allignment scores and encoder hidden states
    context_infer = dot([attention, decoder_context_inputs], axes=[2, 1])

    # Concatenate context vector and hidden decoder states
    decoder_context_outputs = concatenate([context_infer, decoder_outputs])

    # Get the decoder predictions
    decoder_outputs = decoder_dense(decoder_context_outputs)

    decoder_model = Model([decoder_inputs] + [decoder_context_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return decoder_model

# Generate sequence
def generate_sequence(input_sequence, encoder_model, decoder_model):
    # Encode the input sequence
    states_value = encoder_model.predict(input_sequence, verbose=False)

    # Initialize the target sequence with a start token
    target_sequence = np.zeros((1, 1, 1))
    target_sequence[0, 0, :] = w2v.wv.key_to_index["<sos>"]
    START_TOKEN = w2v.wv.key_to_index["<sos>"]
    END_TOKEN = w2v.wv.key_to_index["<eos>"]
    # Generate the output sequence
    output_sequence = []
    end_condition = False
    i = 0
    while i < MAXIMUM_LENGTH and not end_condition:
        i+=1
        decoder_output, state_hidden, state_cell = decoder_model.predict([target_sequence] + states_value, verbose=False)
        output_token = decoder_output[0, -1, :]
        index = np.argmax(output_token)
        if index == END_TOKEN:
            end_condition = True
        word = w2v.wv.index_to_key[index]
        output_sequence.append(word)
        target_sequence = np.zeros((1, 1, 1))
        target_sequence[0, 0, :] = index
        # Update the states value
        states_value = [state_hidden, state_cell]

    return np.array(output_sequence)

def predict(test, train, w2v):
    maximum_length_output = np.max([len(d) for d in train["Answer"]])
    vocab = w2v.wv
    
    predictions = []

    for idx, line in test.iterrows():
        if (idx % 100 == 0):     
            print("progress:", idx/len(test))
            
        words_input_indices = [(w2v.wv.key_to_index[w] if w in vocab else 0) for w in line["Question"]]
        words_input_indices_padded = np.pad(words_input_indices, (0, maximum_length_output - len(words_input_indices)))
        # Example input sequence
        input_sequence = words_input_indices_padded[np.newaxis, :]
        # Generate the output sequence
        output_sequence = generate_sequence(input_sequence)
        predictions.append(output_sequence)
        
    test["Predicted Answer Baseline"] = predictions
    test.to_csv('results_baseline.csv', index=False)

    # Save the dataframe to a file
    with open('results_baseline.pickle', 'wb') as file:
        pickle.dump(test, file)

    sample_data = test["Question"].sample(20)

    myfile = open('sample_results-baseline.txt', 'w')

    for q in sample_data:
        words_input_indices = [(w2v.wv.key_to_index[w] if w in vocab else 0) for w in q]
        words_input_indices_padded = np.pad(words_input_indices, (0, maximum_length_output - len(words_input_indices)))
        # Example input sequence
        input_sequence = words_input_indices_padded[np.newaxis, :]
        # Generate the output sequence
        output_sequence = generate_sequence(input_sequence)
        sequence = ""
        for word in q:
            sequence += " " + word
        print("Question:" + sequence)
        myfile.writelines(sequence + "\n")
        sequence = ""
        for word in output_sequence:
            sequence += " " + word
        print("Answer:" + sequence)
        myfile.writelines(sequence + "\n")

    myfile.close

def main():
    train = get_data('train.csv')
    test = get_data('test.csv')

    sentences = train['Question'] + train['Answer']
    w2v = Word2Vec(sentences=sentences, min_count=1, vector_size=EMBEDDING_DIM, workers=8)
    model = build_model()
    predict(test, train, w2v)
