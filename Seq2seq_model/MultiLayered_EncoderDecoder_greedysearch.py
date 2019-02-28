#for multilayered

#feature_vector_size=100
number_of_hidden_units=10
#vocab=[0]*10000


#encoder architecture
enc_input=Input(shape=(None,feature_vector_size))
enc_lstm_1=LSTM(number_of_hidden_units,return_sequences=True,return_state=True)
enc_lstm_2=LSTM(number_of_hidden_units,return_sequences=True,return_state=True)
enc_lstm_3=LSTM(number_of_hidden_units,return_sequences=False,return_state=True)
x_enc_1,state_h_enc_1,state_c_enc_1=enc_lstm_1(enc_input)
x_enc_2,state_h_enc_2,state_c_enc_2=enc_lstm_2(x_enc_1)
x_enc_3,state_h_enc_3,state_c_enc_3=enc_lstm_3(x_enc_2)
state_enc_1=[state_h_enc_1,state_c_enc_1]
state_enc_2=[state_h_enc_2,state_c_enc_2]
state_enc_3=[state_h_enc_3,state_c_enc_3]
state_enc=state_enc_1+state_enc_2+state_enc_3

#decoder architecture
dec_input=Input(shape=(None,feature_vector_size))
dec_lstm_1=LSTM(number_of_hidden_units,return_sequences=True,return_state=True)
dec_lstm_2=LSTM(number_of_hidden_units,return_sequences=True,return_state=True)
dec_lstm_3=LSTM(number_of_hidden_units,return_sequences=True,return_state=True)
dec_dense=Dense(len(vocab), activation='softmax')
x_dec_1,_,_=dec_lstm_1(dec_input, initial_state=state_enc_1)
x_dec_2,__,__=dec_lstm_2(x_dec_1, initial_state=state_enc_2)
x_dec_3,___,___=dec_lstm_3(x_dec_2, initial_state=state_enc_3)
dec_final_output  = dec_dense(x_dec_3)
model=Model([enc_input,dec_input],dec_final_output)
rms=RMSprop(lr=0.001, decay=1e-8)
model.compile(optimizer=rms,loss='categorical_crossentropy')
model.summary()


#for multilayered

#encoder inference model
model_enc=Model(enc_input,[x_enc_3]+state_enc_1+state_enc_2+state_enc_3)

#decoder_inference_model
decoder_input_state_1_h=Input(shape=(number_of_hidden_units,))
decoder_input_state_1_c=Input(shape=(number_of_hidden_units,))
decoder_input_state_2_h=Input(shape=(number_of_hidden_units,))
decoder_input_state_2_c=Input(shape=(number_of_hidden_units,))
decoder_input_state_3_h=Input(shape=(number_of_hidden_units,))
decoder_input_state_3_c=Input(shape=(number_of_hidden_units,))

decoder_input_state_1=[decoder_input_state_1_h,decoder_input_state_1_c]
decoder_input_state_2=[decoder_input_state_2_h,decoder_input_state_2_c]
decoder_input_state_3=[decoder_input_state_3_h,decoder_input_state_3_c]

output_decoder_1, state_1_h, state_1_c=dec_lstm_1(dec_input,initial_state=decoder_input_state_1)
decoder_state_1=[state_1_h,state_1_c]
output_decoder_2, state_2_h, state_2_c=dec_lstm_2(output_decoder_1,initial_state=decoder_input_state_2)
decoder_state_2=[state_2_h,state_2_c]
output_decoder_3, state_3_h, state_3_c=dec_lstm_3(output_decoder_2,initial_state=decoder_input_state_3)
decoder_state_3=[state_3_h,state_3_c]
decoder_final_output=dec_dense(output_decoder_3)

model_dec=Model([dec_input]+decoder_input_state_1+decoder_input_state_2+decoder_input_state_3,
               [decoder_final_output] +decoder_state_1+decoder_state_2+decoder_state_3)



def predict_new(sequence):
    output,h_1,c_1,h_2,c_2,h_3,c_3=model_enc.predict(sequence)
    initial_state_1,initial_state_2,initial_state_3=[h_1,c_1],[h_2,c_2],[h_3,c_3]
    target_val=model_preprocess['\t'].reshape(1,1,feature_vector_size)
    translated=''
    stop=False
    while not stop:
        output,h_1,c_1,h_2,c_2,h_3,c_3=model_dec.predict([target_val]+initial_state_1+initial_state_2+
                                                         initial_state_3)
        initial_state_1,initial_state_2,initial_state_3=[h_1,c_1],[h_2,c_2],[h_3,c_3]
        max_index=np.argmax(output[0,-1,:])
        characters=index_to_word_dict[max_index]
        translated+=characters
        if (characters=='\n' or len(translated)>=10):
            stop=True
        target_val=model_preprocess[characters].reshape(1,1,feature_vector_size)
    return translated
