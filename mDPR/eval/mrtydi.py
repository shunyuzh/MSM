import os
import argparse
import prettytable as pt

def read_file(label_file, pred_files, coefs):
    labels, preds = dict(), dict()

    with open(label_file, 'r') as fr:
        for line in fr.readlines():
            s = line.split(' ')
            if s[0] in labels:
                labels[s[0]].append(s[2])
            else:
                labels[s[0]] = [s[2]]
    
    for pred_file, coef in zip(pred_files, coefs):
        with open(pred_file, 'r') as fr:
            for line in fr.readlines():
                s = line.split(' ')
                if s[0] in preds:
                    preds[s[0]][s[2]] = preds[s[0]].get(s[2], 0) + float(s[4]) * coef
                else:
                    preds[s[0]] = {s[2]: float(s[4]) * coef}
    
    new_preds = dict()
    for k, v in preds.items():
        id_scores = [(_k, _v) for _k, _v in v.items()]
        id_scores = sorted(id_scores, key=lambda id_score: id_score[1], reverse=True)

        new_preds[k] = [id_score[0] for id_score in id_scores]
    
    return labels, new_preds


metrics = ['MRR@100', 'Recall@100']


def evaluation_with_max_mrr(labels, preds):

    recall100, mrr100 = list(), list()
    for q, label in labels.items():
        r100, m100 = 0, 0
        for doc in label:
            if q in preds and doc in preds[q]:
                index = preds[q].index(doc)

                if index < 100:
                    r100 = r100 + 1
                    m100 = max(m100, (1 / (index + 1)))

        mrr100.append(m100)
        recall100.append(r100 / len(label))

    return {
        'MRR@100': sum(mrr100) / len(mrr100), 
        'Recall@100': sum(recall100) / len(recall100)
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--set', default='test', type=str)
    parser.add_argument('--coef', default=None, nargs='+', type=float)
    parser.add_argument('--pred_file', required=True, nargs='+', type=str)
    parser.add_argument('--label_file', default='data/mrtydi/{}/qrels.{}.txt', type=str)

    args = parser.parse_args()

    if args.coef is None:
        args.coef = [1.0 for _ in args.pred_file]

    nums, results = 0, {m: [] for m in metrics}
    for lang in ['ar', 'bn', 'en', 'fi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th']:
        label_file = args.label_file.format(lang, args.set)
        pred_file = ['{}/{}_{}.tsv'.format(p, args.set, lang) for p in args.pred_file]

        if not os.path.exists(pred_file[0]):
            for m in metrics:
                results[m].append(0)
        else:
            labels, preds = read_file(label_file, pred_file, args.coef)
            result = evaluation_with_max_mrr(labels, preds)
            for m in metrics:
                results[m].append(result[m])
            nums = nums + 1
    
    for m in metrics:
        results[m].append(sum(results[m]) / nums if nums > 0 else 0)
    table = pt.PrettyTable()
    table.field_names = ['metrics', 'ar', 'bn', 'en', 'fi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'avg']
    for k, v in results.items():
        table.add_row([k] + ['%.3f' % s for s in v])
    table.set_style(pt.PLAIN_COLUMNS)
    print(table)


if __name__ == '__main__':
    main()
