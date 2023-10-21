import operator
from collections import defaultdict #different

from utils.clean_arabic import clean_arabic

def lowest_cost_action(ic, dc, sc, im, dm, sm, cost):
    best_action = None
    best_match_count = -1
    min_cost = min(ic, dc, sc)

    if min_cost == sc and cost == 0:
        best_action = 'equal'
    elif min_cost == sc and cost == 1:
        best_action = 'replace'
    elif min_cost == ic and im > best_match_count:
        best_action = 'insert'
    elif min_cost == dc and dm > best_match_count:
        best_action = 'delete'
    
    return best_action


def filter_words(words):
    return filter(filter_word, words)


def filter_word(w):
    return u'[' not in w and u'<' not in w


class SequenceMatcher(object):
    def __init__(self, a=None, b=None, test=operator.eq, action_function=lowest_cost_action):
        if a is None:
            a = []
        if b is None:
            b = []
        
        self.seq1 = list(a)
        self.seq2= list(b)
        self._reset_object()
        self.action_function = action_function
        self.test = test
        self.dist = None
        self._matches = None
        self.opcodes = None

    def _reset_object(self):
        self.opcodes = None
        self.dist = None
        self._matches = None

    def get_opcodes(self):
        if not self.opcodes:
            d, m, opcodes = edit_distance_backpointer(self.seq1, self.seq2, action_function=self.action_function, test=self.test)

            if self.dist:
                assert d == self.dist
            if self._matches:
                assert m == self._matches
            self.dist = d
            self._matches = m
            self.opcodes = opcodes
        return self.opcodes
    
def edit_distance_backpointer(seq1, seq2, action_function=lowest_cost_action, test=operator.eq):
    matches = 0
    m = len(seq1)
    n = len(seq2)
    #distances array
    d = [[0 for x in range(n+1)] for y in range(m+1)]
    #breakpointer array
    bp = [[None for x in range(n+1)] for y in range(m+1)]
    #matches
    matches = [[0 for x in range(n+1)] for y in range(m+1)]

    for i in range(1, m+1):
        d[i][0] = i
        bp[i][0] = ['delete', i -1, i, 0,0]

    for j in range(1, n+1):
        d[0][j] = j
        bp[0][j] = ['insert', 0,0, j-1, j]
    # compute the edit distance
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if test(seq1[i-1], seq2[j-1]) else 1
            ins_cost = d[i][j-1] +1
            del_cost = d[i-1][j] + 1
            sub_cost = d[i-1][j- 1] + cost

            ins_match = matches[i][j-1]
            del_match = matches[i-1][j]
            sub_match = matches[i-1][j-1] + int(not cost)

            action  = action_function(ins_cost, del_cost, sub_cost, ins_match, del_match, sub_match, cost)

            if action == 'equal':
                d[i][j] = sub_cost
                matches[i][j] = sub_match
                bp[i][j] = ['equal', i-1, i, j-1, j]
            elif action == 'replace':
                d[i][j] = sub_cost
                matches[i][j] = sub_match
                bp[i][j] = ['replace', i-1, i, j-1, j]
            elif action == 'insert':
                d[i][j] = ins_cost
                matches[i][j] = ins_match
                bp[i][j] = ['insert', i -1, i-1, j-1, j]
            elif action == 'delete':
                d[i][j] = del_cost
                matches[i][j] = del_match
                bp[i][j] = ['delete', i-1, i, j-1, j-1]
            else:
                raise Exception('Invalid')
            
    opcodes = get_opcodes_from_bp_tables(bp)
    return d[m][n], matches[m][n], opcodes
    

def get_opcodes_from_bp_tables(bp):
    x = len(bp) - 1
    y = len(bp[0]) - 1
    opcodes = []
    while x != 0 or y != 0:
        this_bp = bp[x][y]
        opcodes.append(this_bp)
        if this_bp[0] == 'equal' or this_bp[0] == 'replace':
            x = x -1
            y = y - 1

        elif this_bp[0] == 'insert':
            y = y - 1
        elif this_bp[0] == 'delete':
            x = x - 1
    opcodes.reverse()
    return opcodes


# Verify wer
DELETION_BAR = 100
higher_deletion_indices = []

def percent(ref, num):
    return 100 * (float(num) / float(ref))

def compute_wer(ref, hyp):
    ref_words = ref.split(' ')
    hyp_words = hyp.split(' ')

    sm = SequenceMatcher(a = filter_words(ref_words), b = filter_words(hyp_words))
    ops = defaultdict(int)

    for op in sm.get_opcodes():
        ops[op[0]] += 1

    deletes = ops.get('delete', 0)
    inserts = ops.get('insert', 0)
    subs = ops.get('replace', 0)
    matches = ops.get('equal',  0)

    n_ref_words = len(ref_words)
    if n_ref_words == 0:
        n_ref_words = 0.01

    wer = percent(n_ref_words, deletes+inserts+subs)

    return {
        'matches': percent(n_ref_words, matches),
        'subs': percent(n_ref_words, subs),
        'inserts': percent(n_ref_words, inserts),
        'deletes': percent(n_ref_words, deletes),
        'wer': wer,
        'n_ref_words': n_ref_words,
        'changes': deletes + inserts + subs
    }

def compute_mean_wer(refs, hyps):
    n = 0
    s = 0.0

    for i in range(len(refs)):
        ref = refs[i]
        hyp = hyps[i]

        n += 1
        s += compute_wer(ref, hyp)['wer']

    return s/n

def compute_wer_no_punctuation(ref, hyp):
    return compute_wer(clean_arabic(ref.strip()), clean_arabic(hyp.strip()))


def compute_mean_wer_no_punctuation(refs, hyps):
    n = 0
    s = 0.0

    for i in range(len(refs)):
        ref = refs[i]
        hyp = hyps[i]

        n += 1
        s += compute_wer(clean_arabic(ref.strip()), clean_arabic(hyp.strip()))['wer']

    return s/n


if __name__ == '__main__':
    print(compute_wer(['a', 'b', 'c'], ['a', 'b']))