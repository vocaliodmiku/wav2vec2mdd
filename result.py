import argparse
import os 
import pickle as pkl
from tqdm import tqdm
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "typ", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "ckpt", metavar="DIR", help="root directory containing flac files to index"
    )
    return parser

class Result():
    def __init__(self, annotation, reference, wav_ids, prediction):
        self.annotation = annotation
        self.reference = reference
        self.wav_ids = wav_ids
        self.prediction = prediction
        self.data = self.merge_data() 

    @classmethod
    def setup(cls, data_path, _type, ckpt):

        # reference = pkl.load(open(os.path.join(data_path, 'test.ref'), 'rb'))
        _reference  = open(os.path.join(data_path, _type + '.ref'), 'r').readlines()
        wav_ids = open(os.path.join(data_path, _type + '.tsv'), 'r').readlines()[1:]
        def get_wav_id(x):
            prefix = x.split('\t')[0].split('/')[-3]
            postfix = x.split('\t')[0].split('/')[-1].split('.')[0]
            return "%".join([prefix, postfix])
        wav_ids = list(map(get_wav_id, wav_ids))
        reference  = {}
        for i, j in zip(wav_ids, _reference):
            reference[i] = j.strip('\n').split(' ')
        prediction = open('data/hypo.units-{}-{}.txt'.format(ckpt.split('/')[-1], _type), 'r').readlines()
        annotation = open('data/ref.units-{}-{}.txt'.format(ckpt.split('/')[-1], _type), 'r').readlines()
        return cls(annotation, reference, wav_ids, prediction)

    def merge_data(self):
        prediction = {}
        for i in self.prediction:
            i = i.strip().split(' ')
            a = i[-1].strip('()').split('-')
            wav_id = self.wav_ids[int( i[-1].strip('()').split('-')[-1] )]
            prediction[ wav_id ] = i[:-1]

        annotation = {}
        for i in self.annotation:
            i = i.strip().split(' ')
            wav_id = self.wav_ids[int( i[-1].strip('()').split('-')[-1] )]
            annotation[ wav_id ] = i[:-1]
        data = {}
        for wav_id in self.wav_ids:
            data[wav_id] = {'ref': self.reference[wav_id], 'anno': annotation[wav_id], 'hypo':  prediction[wav_id]}
        return data

    def align(self):
        with open('result/ref.txt', 'w') as ref_f, open('result/annotation.txt', 'w') as ann_f, open('result/hypo.txt', 'w') as hyp_f:
            for wav_id in tqdm(self.wav_ids):
                ann, hyp, ref= self.data[wav_id]['anno'], self.data[wav_id]['hypo'], self.data[wav_id]['ref']
                print("{} {}".format(wav_id, " ".join([i for i in ref if not ( i == 'sil' or i == 'sp')])), file=ref_f)
                print("{} {}".format(wav_id, " ".join([i for i in ann if not ( i == 'sil' or i == 'sp')])), file=ann_f)
                print("{} {}".format(wav_id, " ".join([i for i in hyp if i != 'sil'])), file=hyp_f)

        pass

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    result_data = Result.setup(args.path, args.typ, args.ckpt)
    result_data.align()
