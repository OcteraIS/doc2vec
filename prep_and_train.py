import smart_open
import pandas as pd
import logging 

from gensim.models import doc2vec
import gensim
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

from timeit import default_timer as timer
import model 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

answers_file = '/Users/nernst/Documents/projects/design-detect/sotorrent/answers-only.txt'
questions_file = '/Users/nernst/Documents/projects/design-detect/sotorrent/questions-only.txt'
brunet_file = '/Users/nernst/Documents/projects/design-detect/minEval/data/brunet.csv'
satd_file = '/Users/nernst/Documents/projects/design-detect/tse.satd.data/dataset/technical_debt_dataset.csv'
so_nondesign_file = '/Users/nernst/Documents/projects/design-detect/sotorrent/sotorrent_nondesign_qs.csv'
combined_file = '/Users/nernst/Documents/projects/design-detect/sotorrent/combined.csv'

def read_corpus(fname, tokens_only=False,line_only=False): 
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if line_only: 
                yield line
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def read_txt(fname):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for line in f:
            yield line

def clean_text(text,clean_html=False):
    #http://bdewilde.github.io/blog/blogger/2013/04/16/intro-to-natural-language-processing-2/
    import re
    from bs4 import BeautifulSoup
    if clean_html:
        soup = BeautifulSoup(text)
        [s.extract() for s in soup('code')] # remove <code> and contents
        text = soup.get_text() # get rid of other html junk
    # remove digits with regular expression
    text = re.sub(r'\d', ' ', text)
    # remove any patterns matching standard url format
    url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    text = re.sub(url_pattern, ' ', text)
    # remove all non-ascii characters
    text = ''.join(character for character in text if ord(character)<128)
    # standardize white space
    text = re.sub(r'\s+', ' ', text)
    # drop capitalization
    text = text.lower()
    return text

def clean_data():
    ones = pd.DataFrame([1]*len(answers_list))
    # design answers from SO
    answers_list = list(read_txt(answers_file))
    answers_list = [clean_text(text) for text in answers_list]
    answers_df = pd.DataFrame(answers_list)
    answers_df['label'] = ones
    answers_df.columns = ['discussion','label']
    # answers_df

    # 25k SO questions NOT tagged design, as negative labels
    negative_df = pd.read_csv(so_nondesign_file)
    negative_df = [clean_text(text,clean_html = True) for text in negative_df.iloc[:,1]]
    negative_df = pd.DataFrame(negative_df)
    negative_df['label'] = pd.DataFrame([0]*len(satd_df))
    negative_df.columns = ['discussion','label']

    # join with positive examples
    combined_df = negative_df.append(answers_df)
    combined_df.to_csv(combined_file,index=False)

def compare_by_eye():
    # TODO fix this so it properly references the IDs from the embedding    
    train_corpus = list(read_corpus(combined_file))
    test_corpus = list(read_corpus(brunet_file, tokens_only=True))
    doc_id = random.randint(0, len(test_corpus) - 1)
    inferred_vector = d2v.infer_vector(test_corpus[doc_id])
    sims = d2v.docvecs.most_similar([inferred_vector], topn=len(d2v.docvecs))
    #print(train_corpus[2])
    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    # print(label, sims[0])
    print('SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        # index = int(index[6:])
        doc_id = int(sims[index][0][6:])
        print('%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[doc_id].words)))

if __name__ == "__main__":
    fname = 'd2v.model'
    brunet = pd.read_csv(brunet_file)
    d2v = doc2vec.Doc2Vec.load(fname)
    compare_by_eye()
    # x_train, x_test, y_train, y_test, all_data = model.read_dataset('/Users/nernst/Documents/projects/design-detect/sotorrent/combined.csv')
    # classifier = model.train_classifier(d2v,x_train,y_train)
    # # replace test with brunet
    # model.test_classifier(d2v,classifier,brunet.discussion, brunet.label)
    # # now try zeroR
    # clf = DummyClassifier(strategy='most_frequent', random_state=0)
    # clf.fit(x_train, y_train) #??

    # clf.score(brunet.discussion, brunet.label)

