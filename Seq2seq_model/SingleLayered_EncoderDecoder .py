
"""This is the architecture of the model that is going to be trained.
"""
num_hidden_units=1024

input_enc=Input(shape=(None,feature_vector_size))
lstm_enc=LSTM(num_hidden_units,return_state=True,return_sequences=False,dropout=0.2) #encoder
out_enc,state_enc_h,state_enc_c=lstm_enc(input_enc)
enc_states=[state_enc_h,state_enc_c]

input_dec=Input(shape=(None,feature_vector_size))
lstm_dec=LSTM(num_hidden_units,return_sequences=True,return_state=True,dropout=0.2)
out_dec,_,_=lstm_dec(input_dec,initial_state=enc_states)  #decoder
out_dense_dec_1=Dense(512,activation='relu')
out_dense_dec_2=Dense(len(vocab),activation='softmax')
out_dec=out_dense_dec_1(out_dec)
out_dec=out_dense_dec_2(out_dec)

model=Model([input_enc,input_dec],out_dec)
rms=Adam(lr=0.001, decay=1e-8)

model.compile(optimizer=rms,loss='categorical_crossentropy')
model.summary()

"""These are the encoder and decoder inference models, these models use the pretrained layers
from the the previous model and use it to predict  the response of a new sequence."""\

#for predicting on new input sentences after the model has been trained
model_enc=Model(input_enc,enc_states)

decoder_input_state_h=Input(shape=(num_hidden_units,))
decoder_input_state_c=Input(shape=(num_hidden_units,))
decoder_input_states=[decoder_input_state_h,decoder_input_state_c]
output_dec,state_h,state_c=lstm_dec(input_dec,initial_state=decoder_input_states)
decoder_states=[state_h,state_c]
output_dec=out_dense_dec_1(output_dec)
output_dec=out_dense_dec_2(output_dec)
model_dec=Model(inputs=[input_dec]+decoder_input_states,outputs=[output_dec]+decoder_states) #mind that input
#input should come from an Input layer only , here in the input we had to initialize two Input layer to
#pass states as input.

def predict_new_single_layer(input_sequence):
    """
      Input: a input sentence from the user, where each word has been converted into
      its vector form, and the sentence has been padded to max_question_length.
    """
    initial_val=model_enc.predict(input_sequence)
    target_val=model_preprocess['\t'].reshape(1,1,feature_vector_size)
    translated=''
    stop=False

    while not stop:
        output,h,c=model_dec.predict([target_val]+initial_val)
        initial_val=[h,c]
        max_val_index=np.argmax(output[0,-1,:])
        characters=index_to_word_dict[max_val_index]
        translated+=characters+" "
        if (characters=='\n' or len(translated)>=10):
            stop=True
        target_val=model_preprocess[characters].reshape(1,1,feature_vector_size)
    return translated
