flag_answer_length_upper,flag_answer_length_lower= 15, 3
flag_question_length_upper,flag_question_length_lower= 20, 3


question_preprocess_flagged=[ques for ques, ans in zip (question_preprocess,answer_preprocess)
                             if len(ans)<=flag_answer_length_upper and len(ans)>=flag_answer_length_lower
                             and len(ques)<=flag_question_length_upper and len(ques)>=flag_question_length_lower]
answer_preprocess_flagged=[ans for ques, ans in zip (question_preprocess,answer_preprocess)
                             if len(ans)<=flag_answer_length_upper and len(ans)>=flag_answer_length_lower
                             and len(ques)<=flag_question_length_upper and len(ques)>=flag_question_length_lower]


def vocab_count(corpus):
    """
This function is for obtaining the occurence frequency of each word in the overall
corpus, i.e including both questions and answers.
    """

    flatten_sequence=[i for j in corpus for i in j]
    vocab_counter=Counter(flatten_sequence)
    return vocab_counter
vocab_count_dict=vocab_count(question_preprocess_flagged+answer_preprocess_flagged)

def replace_less_frequent(sequences,minimun_count):
""" This function replaces the less frequent words, with occurence frequency less than
minimum_count, in the corpus with "unk" token.
"""

    for i in range(len(sequences)):
        for index,word in enumerate(sequences[i]):
            if vocab_count_dict[word]<minimun_count:
                sequences[i][index]='unk'
replace_less_frequent(question_preprocess_flagged,3)
replace_less_frequent(answer_preprocess_flagged,3)

def word2vec(vector_size,corpus,min_count):
    """
    Inputs:vector_size-->Desired size of the word embedding
           corpus-->Corpus on which the Word2Vec model is to be trained
           min_count-->Words that occur less than this are excluded while training the model
    Output: a trained word2vec model.
    """
    model=Word2Vec(corpus,size=vector_size,min_count=min_count)
    return model
feature_vector_size=50
complete_corpus=question_preprocess_flagged+answer_preprocess_flagged
model_preprocess=word2vec(feature_vector_size,complete_corpus,1)


model_answer=Word2Vec(answer_preprocess_flagged,size=10,min_count=1)
vocab=list(model_answer.wv.vocab) #vocab corresponding only to the words that occur in answer.
print (len(vocab))
word_to_index_dict={ch:k for k,ch in enumerate(vocab)}
index_to_word_dict={k:ch for k,ch in enumerate(vocab)}
word_to_one_hot_dict={voc:vec for voc,vec in zip(vocab,to_categorical(range(len(vocab))))}





def final_model_input(question_input,answer_input):
    """
       Inputs: question_input--> list containing the questions.
                answer_input-->list containing the answers.

       Output: We will now have two lists corresponding to encoder and decoder inputs,each element of the two
               lists is a sentence broken into words, and these words are represented as a vectors obtained from
               the word2vec model

    """
    tokenized_question = []
    tokenized_answer = []
    for i in range(number_of_samples):
        q_empty=[]
        a_empty=[]
        for k,ch in enumerate(question_input[i]):
            tok_q=model_preprocess[ch]
            q_empty.append(tok_q)

        for k,ch in enumerate(answer_input[i]):
            tok_a=model_preprocess[ch]
            a_empty.append(tok_a)
        tokenized_question.append(np.asarray(q_empty))
        tokenized_answer.append(np.asarray(a_empty))

    return tokenized_question,tokenized_answer

    encoder_input,decoder_input= final_model_input(question_preprocess_flagged,answer_preprocess_flagged)


    def final_target_data(answers):
    """This fumction is for generating target data.
    Input: List containing answers,where each answer is a list of words, appended by '\t' and ends with '\n'.

    Output: A list containing answers whrere each word is converted into its one hot encoded form,
    also in this list each answer is one time step ahead compared to the decoder input.
    """
    tar_data = [0]*(len(answers))
    for ind_sent in range(len(answers)):
        sent_encoded = [0]*(len(answers[ind_sent])-1)
        sent_word = answers[ind_sent]
        for ind in range(1,len(sent_word)):
            sent_encoded[ind-1]=word_to_one_hot_dict[sent_word[ind]]
        tar_data[ind_sent] = sent_encoded

target_data=final_target_data(answer_preprocess_flagged)

#n_batches=number_of_samples//batch_size
def data_generator(number_of_samples):
    """
    This is a generator function generates data batch wise.

    Input: total number of samples to be fed while training the neural net.

    Output:Yields a batch wise, padded encoder_input,decoder_input and padded target_data.

    """

    batch_size=50
    n_batches=number_of_samples//batch_size

    counter=0
    while 1:
        X_enc=sequence.pad_sequences(encoder_input[counter*batch_size:(counter+1)*batch_size],maxlen=max_question_length,dtype='float32')
        X_dec=sequence.pad_sequences(deocder_input[counter*batch_size:(counter+1)*batch_size],maxlen=max_answer_length,dtype='float32')
        y    =sequence.pad_sequences(target_data[counter*batch_size:(counter+1)*batch_size],maxlen=max_answer_length)
        counter+=1
        yield [X_enc,X_dec],y

        if counter>=n_batches:
            counter=0
