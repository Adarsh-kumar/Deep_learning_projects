
#loading the cornell movie dataset
import os
os.chdir('/home/yash/Downloads/')

class movie_corpus_preprocess:
        def __init__(self,lines,conversations):
            self.movie_lines_filepath = lines
            self.movie_conversations = conversations

        def id2line(self):
            #creates a dictionary to store the person id as keys and text as value
            id2line = {}
            id_index = 0
            text_index = 4
            with open(self.movie_lines_filepath, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    items = line.split(DELIM)
                    if len(items) == 5:
                        line_id = items[id_index]
                        dialog_text = items[text_index]
                        id2line[line_id] = dialog_text
            return id2line
        #to extract converstation ids
        def get_conversations(self):
            conversation_ids_index = -1
            conversations = []
            with open(self.movie_conversations, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    #print(line)
                    items = line.split(DELIM)
                    conversation_ids_field = items[conversation_ids_index]
                    conversation_ids = literal_eval(conversation_ids_field)  # evaluate as a python list
                    conversations.append(conversation_ids)
            return conversations

        def get_question_answer_list(self,id2line,conversation):
            questions=[]
            answers=[]
            count=0
            for a in conversation:
                if count<1000000:
                    #count+=1
                    if len(a)%2!=0:
                        a=a[:-1]
                    for idx, line_id in enumerate(a):
                        if idx % 2 == 0 and line_id:
                            #for ids in line_id:
                #print(line_id)
                                #print(ids)
                            questions.append(id2line[line_id])
                        if idx%2!=0 and  line_id:
                            #for ids in line_id:
                            answers.append(id2line[line_id])
            return questions,answers



movie_lines_filepath='movie_lines.txt'
movie_conversations_filepath='movie_conversations.txt'

movie=movie_corpus_preprocess(movie_lines_filepath,movie_conversations_filepath)
id2line=movie.id2line()
conversation=movie.get_conversations()
movie_question,movie_answer=movie.get_question_answer_list(id2line,conversation)
#movie_question contains questions as simple sentences (not tokenized yet) and
# similarly for movies_answer




#loading scotus dataset
class ScotusData:
    """
    """

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        self.lines = self.loadLines(os.path.join(dirName, "scotus"))
        self.conversations = [{"lines": self.lines}]


    def loadLines(self, fileName):
        """
        Args:
            fileName (str): file to load
        Return:
            list<dict<str>>: the extracted fields for each line
        """
        lines = []

        with open(fileName, 'r') as f:
            for line in f:
                l = line[line.index(":")+1:].strip()  # Strip name of speaker.

                lines.append({"text": l})

        return lines


    def getConversations(self):
        return self.conversations

scotus_data=ScotusData('/home/yash/Desktop/KF_ubuntu/')
scotus_lines=scotus_data.loadLines('/home/yash/Desktop/KF_ubuntu/scotus')
scotus_question=[scotus_lines[i]['text'] for i in range (0,len(scotus_lines),2)]
scotus_answer=[scotus_lines[i]['text'] for i in range(1,len(scotus_lines),2)]




def decontracted(phrase):
    """Input= individual sentences as a single string not broken into words yet
    (eg :'Right.  See?  You are ready for the quiz.')

    Output= patterns such as "you're" will be converted to "you are"
    """

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

question_preprocess=[]
answer_preprocess=[]

def break_into_words(question, answer):
    """
        Input:
            question-->list of sentences that exist as questions for scotus and cornell dataset e.g: movie_question
            answer-->list of sentences that exist as answer for scotus and cornell dataset e.g: movie_answer
        """
    for q,a in zip(question,answer):
        q_new=text_to_word_sequence(expand_short_forms(q)) #we simulataneously expand the short forms.
        a_new=text_to_word_sequence(expand_short_forms(a))
        a_new.append('\n')                                 #adding '\n' as end of sentence token
        a_new= ['\t']+ a_new                                 #adding '\t' as start of sentence
        question_preprocess.append(q_new)
        answer_preprocess.append(a_new)
break_into_words(movie_question,movie_answer)
break_into_words(scotus_question,scotus_answer)

"""Till this point all we have is a list of sequences questions(question_preprocess) in which
each element is a list of words, and similarly a list answer_preprocess.
