import os
import gc
from flask import Flask, request, jsonify
import pickle
import re
from collections import Counter
from google.cloud import storage
import math
import nltk

# ============== Configuration ==============
PROJECT_ID = 'hw3ir-480513'
BUCKET_NAME = 'dotan-irassignment3-bucket'

# ============== Tokenizer ==============
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [tok for tok in tokens if tok not in all_stopwords]


# ============== GCS Helpers ==============
def get_bucket():
    client = storage.Client.create_anonymous_client()
    return client.bucket(BUCKET_NAME)


def load_pickle_from_gcs(filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from GCS...")
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        bucket = get_bucket()
        blob = bucket.blob(filename)
        blob.download_to_filename(filename)
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_local_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# ============== Multi-File Reader ==============
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6


class MultiFileReader:
    def __init__(self, bucket):
        self._bucket = bucket
        self._cache = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._cache:
                blob = self._bucket.blob(f_name)
                self._cache[f_name] = blob.download_as_bytes()
            data = self._cache[f_name]
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(data[offset:offset + n_read])
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        self._cache.clear()


def read_posting_list(index, term, bucket):
    if term not in index.posting_locs:
        return []
    reader = MultiFileReader(bucket)
    locs = index.posting_locs[term]
    b = reader.read(locs, index.df[term] * TUPLE_SIZE)
    posting_list = []
    for i in range(index.df[term]):
        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
        tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
        posting_list.append((doc_id, tf))
    reader.close()
    return posting_list


# ============== Load Data ==============
print("Loading data...")

body_index = load_pickle_from_gcs('postings_gcp/index.pkl')
gc.collect()
print("Body index loaded.")

doc_lengths = load_pickle_from_gcs('doc_lengths.pkl')
gc.collect()
print("Doc lengths loaded.")

doc_titles = load_pickle_from_gcs('doc_titles.pkl')
gc.collect()
print("Doc titles loaded.")

pagerank = load_pickle_from_gcs('pagerank.pkl')
gc.collect()
print("PageRank loaded.")

title_postings = load_local_pickle('title_postings.pkl')
print(f"Title postings loaded: {len(title_postings)} terms")

try:
    pageviews = load_pickle_from_gcs('pageviews.pkl')
    print("Pageviews loaded.")
except:
    pageviews = {}
    print("Pageviews not found.")

# Pre-calculate constants
N = len(doc_lengths)
avg_doc_len = sum(doc_lengths.values()) / N if N > 0 else 1
bucket = get_bucket()

print(f"Total documents: {N}, Average doc length: {avg_doc_len:.2f}")
print("All data loaded!")


# ============== Flask App ==============
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def get_bm25_scores(query_tokens, k1=1.5, b=0.75):
    """Calculate BM25 scores."""
    scores = Counter()
    
    for term in query_tokens:
        if term not in body_index.df:
            continue
        posting_list = read_posting_list(body_index, term, bucket)
        df = body_index.df[term]
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        for doc_id, tf in posting_list:
            if doc_id not in doc_lengths:
                continue
            doc_len = doc_lengths[doc_id]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            scores[doc_id] += idf * (numerator / denominator)
    
    return scores


def get_title_scores(query_tokens_set):
    """Get title matching scores with term count."""
    title_scores = Counter()
    title_match_count = Counter()
    
    for term in query_tokens_set:
        if term in title_postings:
            # IDF for this term
            df = len(title_postings[term])
            idf = math.log(N / (df + 1))
            for doc_id in title_postings[term]:
                title_scores[doc_id] += idf
                title_match_count[doc_id] += 1
    
    return title_scores, title_match_count


@app.route("/search")
def search():
    """Main search - balanced combination of title and body."""
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify([])

    query_tokens = tokenize(query)
    if not query_tokens:
        return jsonify([])

    query_tokens_set = set(query_tokens)
    num_query_terms = len(query_tokens_set)
    
    # Get scores from both sources
    title_scores, title_match_count = get_title_scores(query_tokens_set)
    bm25_scores = get_bm25_scores(query_tokens)
    
    # Combine all candidate documents
    all_docs = set(title_scores.keys()) | set(bm25_scores.keys())
    
    if not all_docs:
        return jsonify([])
    
    # Normalize scores
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
    max_title = max(title_scores.values()) if title_scores else 1
    
    # Calculate final scores
    final_scores = Counter()
    
    for doc_id in all_docs:
        score = 0
        
        # === TITLE SCORE ===
        if doc_id in title_scores:
            # Normalized title IDF score (0-1)
            norm_title = title_scores[doc_id] / max_title
            
            # How many query terms match in title
            match_count = title_match_count[doc_id]
            match_ratio = match_count / num_query_terms
            
            # Title bonus based on coverage
            if match_ratio >= 0.8:  # Almost all terms
                title_weight = 200
            elif match_ratio >= 0.5:  # Half or more
                title_weight = 100
            elif match_ratio >= 0.3:
                title_weight = 50
            else:
                title_weight = 25
            
            score += norm_title * title_weight
        
        # === BM25 BODY SCORE ===
        if doc_id in bm25_scores:
            norm_bm25 = bm25_scores[doc_id] / max_bm25
            # BM25 gets significant weight, especially when title match is weak
            bm25_weight = 100
            score += norm_bm25 * bm25_weight
        
        # === PAGERANK BOOST ===
        if doc_id in pagerank:
            pr = pagerank[doc_id]
            # Small normalized boost
            score += min(pr * 5, 10)
        
        final_scores[doc_id] = score
    
    # Get top 100
    top_docs = final_scores.most_common(100)
    res = [(doc_id, doc_titles.get(doc_id, "")) for doc_id, score in top_docs]
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """Search using TF-IDF cosine similarity on body text only."""
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify([])

    query_tokens = tokenize(query)
    if not query_tokens:
        return jsonify([])

    query_counter = Counter(query_tokens)
    scores = Counter()

    for term, query_tf in query_counter.items():
        if term not in body_index.df:
            continue

        posting_list = read_posting_list(body_index, term, bucket)
        df = body_index.df[term]
        idf = math.log(N / df) if df > 0 else 0
        query_weight = query_tf * idf

        for doc_id, tf in posting_list:
            if doc_id not in doc_lengths:
                continue
            doc_weight = (tf / doc_lengths[doc_id]) * idf
            scores[doc_id] += query_weight * doc_weight

    top_docs = scores.most_common(100)
    res = [(doc_id, doc_titles.get(doc_id, "")) for doc_id, score in top_docs]
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """Search by title - ALL results ordered by distinct query word matches."""
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify([])

    query_tokens = set(tokenize(query))
    if not query_tokens:
        return jsonify([])

    doc_matches = Counter()
    for term in query_tokens:
        if term in title_postings:
            for doc_id in title_postings[term]:
                doc_matches[doc_id] += 1
    
    sorted_docs = sorted(doc_matches.items(), key=lambda x: (-x[1], x[0]))
    res = [(doc_id, doc_titles.get(doc_id, "")) for doc_id, count in sorted_docs]
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """Search by anchor text."""
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify([])
    return jsonify([])


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """Returns PageRank values for a list of wiki article IDs."""
    wiki_ids = request.get_json()
    if not wiki_ids:
        return jsonify([])
    res = [pagerank.get(int(doc_id), 0.0) for doc_id in wiki_ids]
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """Returns page view counts for a list of wiki article IDs."""
    wiki_ids = request.get_json()
    if not wiki_ids:
        return jsonify([])
    res = [pageviews.get(int(doc_id), 0) for doc_id in wiki_ids]
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
