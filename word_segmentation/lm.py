import collections
import math

SENTENCE_BEGIN = '<s>'
SENTENCE_END = '</s>'

def sliding(xs, window_size):
    for i in range(1, len(xs) + 1):
        yield xs[max(0, i - window_size):i]

def generate_ngrams(xs, window_size):
    ngrams = zip(*[xs[i:] for i in range(window_size)])
    return ngrams if window_size > 1 else (t[0] for t in ngrams)

def remove_all(s, chars):
    return ''.join([c for c in s if c not in chars])

def alpha_only(s):
    s = s.replace('-', ' ')
    return ''.join([c for c in s if c.isalpha() or c == ' '])

def clean_line(l):
    return alpha_only(l.strip().lower())

def words(l):
    return l.split()

############################################################
# Make an n-gram model of words in text from a corpus.

def make_LM(path):
    unigram_counts = collections.Counter()
    total_counts = 0
    bigram_counts = collections.Counter()
    bitotal_counts = collections.Counter()
    VOCAB_SIZE = 600000
    LONG_WORD_THRESHOLD = 5
    LENGTH_DISCOUNT = 0.15

    with open(path, 'r') as f:
        for l in f:
            ws = [SENTENCE_BEGIN] + words(clean_line(l)) + [SENTENCE_END]
            unigrams = list(generate_ngrams(ws, 1))
            bigrams = list(generate_ngrams(ws, 2))
            total_counts += len(unigrams)
            unigram_counts.update(unigrams)
            bigram_counts.update(bigrams)
            bitotal_counts.update([x[0] for x in bigrams])

    def unigram_cost(x):
        if x not in unigram_counts:
            length = max(LONG_WORD_THRESHOLD, len(x))
            return -(length * math.log(LENGTH_DISCOUNT) + math.log(1.0) - math.log(VOCAB_SIZE))
        else:
            return math.log(total_counts) - math.log(unigram_counts[x])

    def bigram_model(a, b):
        return math.log(bitotal_counts[a] + VOCAB_SIZE) - math.log(bigram_counts[(a, b)] + 1)

    return unigram_cost, bigram_model

def log_sum_exp(x, y):
    lo = min(x, y)
    hi = max(x, y)
    return math.log(1.0 + math.exp(lo - hi)) + hi;

def smooth_unigram_and_bigram(unigram_cost, bigram_model, a):
    '''Coefficient `a` is Bernoulli weight favoring unigram'''

    # Want: -log( a * exp(-u) + (1-a) * exp(-b) )
    #     = -log( exp(log(a) - u) + exp(log(1-a) - b) )
    #     = -logSumExp( log(a) - u, log(1-a) - b )

    def smooth_model(w1, w2):
        u = unigram_cost(w2)
        b = bigram_model(w1, w2)
        return -log_sum_exp(math.log(a) - u, math.log(1-a) - b)

    return smooth_model
