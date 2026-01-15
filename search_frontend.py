import builtins
import math
from contextlib import closing
from heapq import nlargest
from flask import Flask, request, jsonify
from pyspark.ml.feature import Tokenizer, RegexTokenizer
import re
from inverted_index_gcp import InvertedIndex, MultiFileReader
from collections import Counter, OrderedDict , defaultdict
import numpy as np
import json
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi



class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

############################################################################################
base_file_path = './'
TUPLE_SIZE=6
num_buck = 800
english_stopwords = frozenset([
    "during", "as", "whom", "no", "so", "shouldn't", "she's", "were", "needn", "then", "on",
    "should've", "once", "very", "any", "they've", "it's", "it", "be", "why", "ma", "over",
    "you'll", "they", "you've", "am", "before", "shan", "nor", "she'd", "because", "been",
    "doesn't", "than", "will", "they'd", "not", "those", "had", "this", "through", "again",
    "ours", "having", "himself", "into", "i'm", "did", "hadn", "haven", "should", "above",
    "we've", "does", "now", "m", "down", "he'd", "herself", "t", "their", "hasn't", "few",
    "and", "mightn't", "some", "do", "the", "we're", "myself", "i'd", "won", "after",
    "needn't", "wasn't", "them", "don", "further", "we'll", "hasn", "haven't", "out", "where",
    "mustn't", "won't", "at", "against", "shan't", "has", "all", "s", "being", "he'll", "he",
    "its", "that", "more", "by", "who", "i've", "o", "that'll", "there", "too", "they'll",
    "own", "aren't", "other", "an", "here", "between", "hadn't", "isn't", "below", "yourselves",
    "ve", "isn", "wouldn", "d", "we", "couldn", "ain", "his", "wouldn't", "was", "didn", "what",
    "when", "i", "i'll", "with", "her", "same", "you're", "yours", "couldn't", "for", "doing",
    "each", "aren", "which", "such", "mightn", "up", "mustn", "you", "only", "most", "of", "me",
    "she", "he's", "in", "a", "if", "but", "these", "him", "hers", "both", "my", "she'll", "re",
    "weren", "yourself", "is", "until", "weren't", "to", "are", "itself", "you'd", "themselves",
    "ourselves", "just", "wasn", "have", "don't", "ll", "how", "they're", "about", "shouldn",
    "can", "our", "we'd", "from", "it'd", "under", "while", "off", "y", "doesn", "theirs",
    "didn't", "or", "your", "it'll"
])
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)

#Loads Inverted Index
with open(base_file_path+"title_index/"+'title_DL.json', 'rt') as f:
  title_DL = json.load(f)
with open(base_file_path+"body_index/"+'body_DL.json', 'rt') as f:
  body_DL = json.load(f)
#with open(base_file_path+"anchor_index/"+'dl_anchor_new.json', 'rt') as f:
    #anchor_DL = json.load(f)

AVGDL_title = sum(float(v) for v in title_DL.values()) / len(title_DL)
AVGDL_body = sum(float(v) for v in body_DL.values()) / len(body_DL)

#Changing posting locs to match local colab file location
#in index creation the index posting location were in the local cluster storage
#therefore we need to update the posting locs to reflect their actual location in the server instance file system

def change_index_locs(index):
    for term,loc_tuple in index.posting_locs.items():
      new_list = []
      for loc,offset in loc_tuple:
        splited = loc.split("/")
        for part in splited:
          if "index" in part:
            rebuilt = part
          if "bin" in part:
            rebuilt = base_file_path + rebuilt + "/" + part
        loc_tuple = (rebuilt,offset)
        new_list.append(loc_tuple)
      index.posting_locs[term] = new_list

title_index=InvertedIndex.read_index(base_file_path+'title_index', 'index')
#anchor_index=InvertedIndex.read_index(base_file_path+'anchor_index', 'index')
body_index=InvertedIndex.read_index(base_file_path+'body_index', 'index')

title_index_stem = InvertedIndex.read_index(base_file_path + 'title_index_stem', 'index')
body_index_stem  = InvertedIndex.read_index(base_file_path + 'body_index_stem', 'index')
#anchor_index_stem = InvertedIndex.read_index(base_file_path + 'anchor_index_stem', 'index')

change_index_locs(body_index)
#change_index_locs(anchor_index)
change_index_locs(title_index)

change_index_locs(title_index_stem)
change_index_locs(body_index_stem)
#change_index_locs(anchor_index_stem)
#

def tokenize(text):
    tokens = []
    for m in RE_WORD.finditer(text.lower()):
        w = m.group()
        if w not in all_stopwords:
            tokens.append(w)
    return tokens

def word_count(text, id):
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  counter={}
  for token in tokens:
    if token not in all_stopwords:
      if token in counter:
        counter[token] += 1
      else:
        counter[token] = 1
  return [(token, (id, tf)) for token, tf in counter.items()]

def reduce_word_counts(unsorted_pl):
  return (sorted(unsorted_pl, key = lambda x: x[0]))

def calculate_df(postings):
  return postings.map(lambda x: (x[0],len(x[1]))).partitionBy(16)

#def token2bucket_id(token):
  #return int(_hash(token),16) % NUM_BUCKETS

def partition_postings_and_write(postings):#####################333
  res = defaultdict(list)
  output_list = []
  bucketed_postings = postings.map(lambda x:(token2bucket_id(x[0]),x)).groupByKey()
  for id,content in bucketed_postings.toLocalIterator():
    output_list.append(InvertedIndex.write_a_posting_list((id,content),location))
  sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
  return sc.parallelize(output_list,numSlices=16)

def find_postings(terms,index):
    res={}
    with closing(MultiFileReader(base_file_path)) as reader:
        for w, locs in index.posting_locs.items():
          if len(res) == len(terms): #if (res.keys() == terms):
            break
          if(w not in terms):
            continue
          else:
            b = reader.read(locs, index.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(index.df[w]):
              doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
              tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
              posting_list.append((doc_id, tf))
            res[w]=posting_list

    return res

def tf_idf(term,doc,DocLen,InvertedIndex,tf):
    tf_norm=tf/DocLen[str(doc)]
    idf=math.log2(len(DocLen)/InvertedIndex.df[term])
    return tf_norm*idf

def cosine_sim(InvertedIndex,DocLen,query,N):
    if not query:
        return []
    dict_count = Counter()
    tf_q = Counter(query) 
    terms = [t for t in tf_q.keys() if t in InvertedIndex.df]
    if not terms:
        return []
 
    pls = find_postings(set(terms), InvertedIndex)
    
    d_tfs = defaultdict(list)
    for t, pl in pls.items():
        for doc_id, tf_d in pl:
            tf_idf_val = tf_idf(t, doc_id, DocLen, InvertedIndex, tf_d)
            dict_count[doc_id] += (tf_q[t]) * tf_idf_val
            d_tfs[doc_id].append(tf_idf_val)
    q_norm = (sum(x * x for x in tf_q.values()) ** 0.5)
    if q_norm == 0:
        return []
    single_term = (len(tf_q) == 1)

    for doc_id, value in dict_count.items():
        if single_term:
            dict_count[doc_id] = value * (1 / q_norm)
        else:
            d_norm = (sum(x * x for x in d_tfs[doc_id]) ** 0.5)
            if d_norm == 0:
                continue
            dict_count[doc_id] = value * (1 / q_norm) * (1 / d_norm)
    return sorted(
        [(doc_id, builtins.round(score, 5)) for doc_id, score in dict_count.items()],
        key=lambda x: x[1],
        reverse=True
    )[:N]

def getCosineSim(query):
    ## returns the 100 doc the had the highest cos sim to the doc
    res = cosine_sim(body_index,body_DL,query,100)
    output = []
    doc_list =  [doc_id for doc_id,rank in res]
    for doc in doc_list:
      output.append((doc,getTitle(doc)))
    return output

def getTitle(id):
  k = id % num_buck
  with open('./doc2title/d2t'+str(k)+'.json', 'rt') as f:
     pr_k = json.load(f)
  for t in pr_k:
    if t[0]==id:
      return t[1]
  return 0

####################################################################################

def text_binary_gnr(query_tokens,dl,index):
    unique_tokenized_query = list(set(query_tokens))
    matches = Counter()
    tfidf_sum = defaultdict(float)
    pls = find_postings(unique_tokenized_query,index)
    for term in unique_tokenized_query:
        for doc_id, freq  in pls.get(term, []):
            matches[doc_id] += 1
            tfidf_sum[doc_id] += tf_idf(term, doc_id, dl, index, freq)
    ranked = sorted(
        matches.items(),
        key=lambda x: (x[1], tfidf_sum[x[0]]),
        reverse=True
    )
    return [(doc_id, getTitle(doc_id)) for doc_id, _ in ranked]

def cos_sim_ranking_body(query_tokens):
    ranked = getCosineSim(query_tokens)
    return [(doc_id, getTitle(doc_id)) for doc_id, _ in ranked]

def binary_ranking_title(query_tokens):
    return text_binary_gnr(query_tokens,title_DL,title_index)

def binary_ranking_anchor(query_tokens):
    return text_binary_gnr(query_tokens,anchor_DL,anchor_index)

############################################################################################

#get the page rank of a single document from the json files on the machine disk
def getPr(id):
  k = id % num_buck
  with open(base_file_path+'page_rank/pr'+str(k)+'.json', 'rt') as f:
     pr_k = json.load(f)
  for t in pr_k:
    if t[0]==id:
      return t[1]
  return 0

def getPv(id):
  k = id % num_buck
  with open(base_file_path+'pv/pv'+str(k)+'.json', 'rt') as f:
    pr_k = json.load(f)
  for t in pr_k:
    if t[0]==id:
      return t[1]
  return 0

def getPageViews(id_list):
    output = []
    for id in id_list:
      output.append(getPv(id))
    return output
############################################################################################

def stemming_tokenize(query):
    tokens = tokenize(query)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def bm25_classic(query_tokens,index,DL,avgdl,k=100,k1=1.2,b=0.75,use_log1p=True):
    if not query_tokens:
        return {}
    N = len(DL)
    if N == 0:
        return {}

    def get_dl(doc_id):
        v = DL.get(str(doc_id), DL.get(doc_id, None))
        return float(v) if v is not None else None

    scores = defaultdict(float)
    for term in set(query_tokens):
        df = index.df.get(term, 0)
        if df <= 0:
            continue
        frac = (N - df + 0.5) / (df + 0.5)
        idf = math.log(1.0 + frac) if use_log1p else math.log(frac)
        posting_list = index.read_a_posting_list(base_file_path, term)  # [(doc_id, tf), ...]
        for doc_id, tf in posting_list:
            dl = get_dl(doc_id)
            if dl is None or dl <= 0:
                continue
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            scores[doc_id] += idf * ((tf * (k1 + 1.0)) / denom)

    if not scores:
        return {}

    return dict(nlargest(k, scores.items(), key=lambda x: x[1]))




def search_engine(query_tokens,title_weight=0.3,body_weight=0.7,N=100):
    title_scores = bm25_classic(query_tokens, title_index_stem,title_DL,AVGDL_title)
    body_scores  = bm25_classic(query_tokens, body_index_stem, body_DL,AVGDL_body)

    merged_scores = {}
    all_docs = set(title_scores.keys()) | set(body_scores.keys())
    for doc_id in all_docs:
        merged_scores[doc_id] = (
            title_weight * title_scores.get(doc_id, 0.0) +
            body_weight  * body_scores.get(doc_id, 0.0)
        )
    top_docs = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:N]
    res = [(int(doc_id), getTitle(doc_id)) for doc_id, _ in top_docs]
    return res

#for playing with the anchor 
def search_engine2(norm_query_tokens,query_tokens,title_weight=0.5,body_weight=0.5,N=100):
    title_scores = bm25_classic(query_tokens, title_index_stem, title_DL,AVGDL_title)
    body_scores  = bm25_classic(query_tokens, body_index_stem,  body_DL,AVGDL_body)

    merged_scores = {}
    all_docs = set(title_scores.keys()) | set(body_scores.keys())
    for doc_id in all_docs:
        merged_scores[doc_id] = (
            title_weight * title_scores.get(doc_id, 0.0) +
            body_weight  * body_scores.get(doc_id, 0.0)
        )
    top_docs = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:N]
    res = [(int(doc_id), getTitle(doc_id)) for doc_id, _ in top_docs]
    return res
############################################################################################
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = stemming_tokenize(query)
    #לסנן מסמכים מועמדים
    res = search_engine(query_tokens)
    # BM25 על מסמכים מועידם שידרג אותם

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    # from pyspark.ml.feature import Tokenizer, RegexTokenizer
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    res = cos_sim_ranking_body(tokenized_query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    res = binary_ranking_title(tokenized_query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    res = binary_ranking_anchor(tokenized_query)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        res.append(getPr(id))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = getPageViews(wiki_ids)
    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
