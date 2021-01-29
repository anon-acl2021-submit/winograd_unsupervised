import sys
import json

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


FOLDS = 50

LANG_LIST = ['en','fr','jp','ru','zh']
features_shift = {
    ('mean','MAS'):-8,
    ('max','MAS'):-6,
    ('mean','normal'):-4,
    ('max','normal'):-2,
}

assert len(sys.argv)>1, 'input filename not found'
assert len(sys.argv)>2, 'output filename not found'
input_filename = sys.argv[1]
output_filename = sys.argv[2]

headers  = ['','base lang','train acc','','valid acc','']
for lang in LANG_LIST: headers.extend( [lang, ''] )

with open(output_filename, 'w', encoding='utf-8') as output_file:
    for penalty, solver in (('l2','lbfgs'), ('l1','liblinear')):
        for normalizer in ('ident', 'diff'):
            for pooling in ('mean', 'max'):
                for direction in ('MAS','normal'):

                    lists_of_best_heads = []

                    for train_lang in LANG_LIST:
                        best_heads = None
                        use_heads = None
                        for heads_number in (0,1,2,4,8,16,32):
                            if heads_number:
                                use_heads = best_heads[:heads_number]

                            X = []
                            y = []
                            X_valids = defaultdict(list)
                            y_valids = defaultdict(list)

                            for line in open(input_filename, encoding='utf-8'):
                                chunks = line.strip().split('\t')
                                lang = chunks[0] 
                                right_answer, decoy = chunks[features_shift[(pooling,direction)]:len(chunks)+features_shift[(pooling,direction)]+2]
                                right_answer = json.loads(right_answer)
                                decoy = json.loads(decoy)
                                if use_heads:
                                    right_answer = [right_answer[idx] for idx in use_heads]
                                    decoy = [decoy[idx] for idx in use_heads]
                                right_answer = np.array(right_answer)
                                decoy = np.array(decoy)

                                if normalizer=='ident':
                                    item1 = right_answer
                                    item2 = decoy
                                else:
                                    item1 = right_answer - decoy
                                    item2 = decoy - right_answer

                                if lang == train_lang:
                                    X.extend( [item1, item2] )
                                    y.extend( [1,0] ) 
                                X_valids[lang].extend( [item1, item2] )
                                y_valids[lang].extend( [1,0] ) 

                            X = np.array(X)
                            y = np.array(y)

                            score_test  = []
                            score_train = []
                            score_valids = defaultdict(list)

                            if not best_heads:
                                weights = defaultdict(float)

                            for seed in range(FOLDS):
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

                                clf = LogisticRegression(random_state=0, penalty=penalty, solver=solver).fit(X_train, y_train)

                                score_train.append( clf.score(X_train, y_train) )
                                score_test.append( clf.score(X_test, y_test) )

                                for lang in LANG_LIST:
                                    score_valids[lang].append( clf.score(np.array(X_valids[lang]), np.array(y_valids[lang])) )
                                if not best_heads:
                                    for q,w in enumerate(clf.coef_[0]):
                                        weights[q] += w

                            if not best_heads:
                                best_heads = [k for k,w in list(sorted(weights.items(), key=lambda x:-abs(x[1])))[:32]]
                                lists_of_best_heads.append( best_heads )
                            res = [np.mean(score_train), np.std(score_train), np.mean(score_test), np.std(score_test)]
                            for lang in LANG_LIST:
                                if lang == train_lang:
                                    res.append('-')
                                    res.append('-')
                                else:
                                    res.append( np.mean(score_valids[lang]) )
                                    res.append( np.std(score_valids[lang]) )

                            dump = [penalty, normalizer, pooling, direction, train_lang]
                            if not use_heads:
                                dump.append('all')
                                dump.append('-')
                            else:
                                suffix = "s" if heads_number>1 else ""
                                dump.append(f'top{heads_number}')
                                dump.append(json.dumps(use_heads))
                            dump.append( '\t'.join(map(str,res)).replace('.',',') )
                            print(f'\t'.join(map(str,dump)), file=output_file)

                    head_score = defaultdict(int)
                    for run in lists_of_best_heads:
                        for idx,i in enumerate(run[::-1]):
                            head_score[i] += idx
                    selected_heads = []
                    for head,_ in sorted(head_score.items(), key=lambda x:-x[1]):
                        selected_heads.append(head)

                    selected_heads = selected_heads[:5]

                    for train_lang in LANG_LIST:
                        use_heads = selected_heads

                        X = []
                        y = []

                        X_valids = defaultdict(list)
                        y_valids = defaultdict(list)

                        for line in open(input_filename, encoding='utf-8'):
                            chunks = line.strip().split('\t')
                            lang = chunks[0] 
                            right_answer, decoy = chunks[features_shift[(pooling,direction)]:len(chunks)+features_shift[(pooling,direction)]+2]
                            right_answer = json.loads(right_answer)
                            decoy = json.loads(decoy)
                            if use_heads:
                                right_answer = [right_answer[idx] for idx in use_heads]
                                decoy = [decoy[idx] for idx in use_heads]
                            right_answer = np.array(right_answer)
                            decoy = np.array(decoy)

                            if normalizer=='ident':
                                item1 = right_answer
                                item2 = decoy
                            else:
                                item1 = right_answer - decoy
                                item2 = decoy - right_answer

                            if lang == train_lang:
                                X.extend( [item1, item2] )
                                y.extend( [1,0] ) 
                            X_valids[lang].extend( [item1, item2] )
                            y_valids[lang].extend( [1,0] ) 

                        X = np.array(X)
                        y = np.array(y)

                        score_test  = []
                        score_train = []
                        score_valids = defaultdict(list)

                        for seed in range(FOLDS):
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

                            clf = LogisticRegression(random_state=0, penalty=penalty, solver=solver).fit(X_train, y_train)
                            score_train.append( clf.score(X_train, y_train) )
                            score_test.append( clf.score(X_test, y_test) )

                            for lang in LANG_LIST:
                                score_valids[lang].append( clf.score(np.array(X_valids[lang]), np.array(y_valids[lang])) )

                        res = [np.mean(score_train), np.std(score_train), np.mean(score_test), np.std(score_test)]
                        for lang in LANG_LIST:
                            if lang == train_lang:
                                res.append('-')
                                res.append('-')
                            else:
                                res.append( np.mean(score_valids[lang]) )
                                res.append( np.std(score_valids[lang]) )

                        dump = [penalty, normalizer, pooling, direction, train_lang, 'best5', json.dumps(selected_heads)]
                        dump.append( '\t'.join(map(str,res)).replace('.',',') )
                        print(f'\t'.join(map(str,dump)), file=output_file)


                    use_heads = selected_heads
                    score_valids = defaultdict(list)

                    for line in open(input_filename, encoding='utf-8'):
                        chunks = line.strip().split('\t')
                        lang = chunks[0] 
                        right_answer, decoy = chunks[features_shift[(pooling,direction)]:len(chunks)+features_shift[(pooling,direction)]+2]
                        right_answer = json.loads(right_answer)
                        decoy = json.loads(decoy)
                        right_score = sum([right_answer[idx] for idx in use_heads])
                        decoy_score = sum([decoy[idx] for idx in use_heads])

                        if right_score>decoy_score:
                            score_valids[lang].append( 1 )
                        else:
                            score_valids[lang].append( 0 )

                    res = ['-', '-', '-', '-']
                    for lang in LANG_LIST:
                        res.append( np.mean(score_valids[lang]) )
                        res.append( '-' )

                    dump = [penalty, normalizer, pooling, direction, 'MAS-like', 'best5', json.dumps(selected_heads)]
                    dump.append( '\t'.join(map(str,res)).replace('.',',') )
                    print(f'\t'.join(map(str,dump)), file=output_file)

