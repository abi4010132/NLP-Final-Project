# imports
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, dot, Activation
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, RMSprop
from keras.utils import split_dataset
import keras_tuner      #!pip install keras-tuner -q

# CONSTANTS
DATASET_SIZE = 90000
MAXIMUM_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 5
EMBEDDING_DIM = 100

# read in data
def get_data():
    data = pd.read_csv('data/single_qna_clean_data.csv', nrows=DATASET_SIZE) # ALL OF IT LETS GO
    data["Question"] = data["Question"].apply(lambda x: eval(x))
    data["Question"] = data["Question"].apply(lambda x: x[:min(MAXIMUM_LENGTH, len(x))])
    data["Question"] = data["Question"].apply(lambda x: x[:len(x) - 1] + ["<eos>"])
    data["Answer"] = data["Answer"].apply(lambda x: eval(x))
    data["Answer"] = data["Answer"].apply(lambda x: x[:min(MAXIMUM_LENGTH, len(x))])
    data["Answer"] = data["Answer"].apply(lambda x: x[:len(x) - 1] + ["<eos>"])
    print("Done with loading")
    return data

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


def custom_loss(y_true, y_pred):
    ssc = SparseCategoricalCrossentropy()
    loss = ssc(y_true, y_pred)
    return loss


def build_model(hp, vocab, embedding_matrix):
    # LSTM units
    dimensionality = hp.Choice("dimensionality", [64, 128, 256, 512])
    num_layers = hp.Choice("num_layers", ["1", "2"])
    
    #Encoder inputs
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(len(vocab), EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(encoder_inputs)
    
    #Decoder inputs
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(len(vocab), EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(decoder_inputs)
    
    with hp.conditional_scope("num_layers", ["1"]):
        if num_layers == "1":
            # Encoder lstm
            encoder_lstm1 = LSTM(dimensionality, return_sequences=True, return_state=True)
            encoder_outputs, state_hidden, state_cell = encoder_lstm1(encoder_embedding)
            encoder_states = [state_hidden, state_cell]
            
            # Decoder lstm
            decoder_lstm1 = LSTM(dimensionality, return_sequences=True, return_state=True)
            decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm1(decoder_embedding, initial_state=encoder_states)
            
    with hp.conditional_scope("num_layers", ["2"]):
        if num_layers == "2":
            encoder_lstm1 = LSTM(dimensionality, return_sequences=True, return_state=False)
            encoder_lstm2 = LSTM(dimensionality, return_sequences=True, return_state=True)
            encoder_outputs, state_hidden, state_cell = encoder_lstm2(encoder_lstm1(encoder_embedding))
            encoder_states = [state_hidden, state_cell]
            
            # Decoder lstm
            decoder_lstm1 = LSTM(dimensionality, return_sequences=True, return_state=False)
            decoder_lstm2 = LSTM(dimensionality, return_sequences=True, return_state=True)
            decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm2(decoder_lstm1(decoder_embedding, initial_state=encoder_states))
    
    
    # Luong Attention 
    # allignment scores are dot product hidden states encoder/decoder
    allignment = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(allignment)

    # context vector is dot product allignment scores and encoder hidden states
    context = dot([attention, encoder_outputs], axes=[2, 1])
    
    # Concatenate context vector and hidden decoder states
    context_decoder_outputs = np.concatenate([context, decoder_outputs])

    # Dense output layer of the model
    decoder_dense = Dense(len(vocab), activation='softmax')
    decoder_outputs = decoder_dense(context_decoder_outputs)
    
    #Model
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    #Compiling
    training_model.compile(optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]), loss=custom_loss, metrics=['accuracy'], sample_weight_mode='temporal')
    return training_model


def main():
    data_all = get_data() 

    data_train = data_all.sample(frac=0.6,random_state=0)
    data_test = data_all.drop(data_train.index)
    data_val = data_test.sample(frac=0.5,random_state=0)
    data_test = data_test.drop(data_val.index)

    embedding_matrix, w2v, vocab = create_embeddings(data_train)
    encoder_input_data_train, decoder_input_data_train, decoder_output_data_train = create_sparse_encoding(w2v, vocab, data_train)
    encoder_input_data_val, decoder_input_data_val, decoder_output_data_val = create_sparse_encoding(w2v, vocab, data_val)
    encoder_input_data_test, decoder_input_data_test, decoder_output_data_test = create_sparse_encoding(w2v, vocab, data_test)

    early_stopping = EarlyStopping(monitor='loss', patience=1)
    callbacks_list = [early_stopping]

    tuner = keras_tuner.GridSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        overwrite=True,
        project_name="attention",
    )
    print(tuner.search_space_summary())

    tuner.search([encoder_input_data_train, decoder_input_data_train], decoder_output_data_train, 
                batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data=([encoder_input_data_val, 
                                decoder_input_data_val], decoder_output_data_val), callbacks=callbacks_list)

    best_model = tuner.get_best_models()[0]

    print(best_model.summary())
    print(tuner.results_summary())

    # Evaluate the model on the test data
    loss, accuracy = best_model.evaluate([encoder_input_data_test, decoder_input_data_test], decoder_output_data_test)

    # Print or use the evaluation results
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)

if __name__ == "__main__":
    main()