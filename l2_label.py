# mix L2-ARCTIC and TIMIT data for MDD
#!/usr/bin/env python3
import argparse
import glob
import os 
import random
import soundfile
import textgrid

import pickle as pkl

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="(Useless) percentage of data to use as validation set (between 0 and 1)"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory",
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument(
        "--seed", default=23333, type=int, metavar="N", help="random seed"
    )
    parser.add_argument(
        "--test-set", nargs='+', type=str, metavar='STRING', help="speakers for test"
    )
    return parser

def parsing_phonesequence(filename):
    phones = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            assert len(line) == 3
            phones.append(line[-1])
    return phones

def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    dir_path = os.path.realpath(args.root)

    ################# L2-ARCTIC  ################# 
    # process annotation files
    search_path = os.path.join(dir_path, "l2/*/annotation/*.TextGrid")
    speakers = []
    l2_data = {}
    l2_phoneset = []
    error_count = 0
    for fname in glob.iglob(search_path, recursive=True):
        try:
            contents = textgrid.TextGrid.fromFile(fname)
        except ValueError:
            continue
        content = [i.mark for i in contents[1] if i.mark != '']
        ann, error_state, ref = split_ref_err(content)
        speaker, sentence = fname.split('/')[-3], fname.split('/')[-1].split('.')[0].split('_')[-1]
        wav_id = speaker + "-" + sentence
        speakers.append(speaker)
        ann, ref = get_phn(ann), get_phn(ref)
        l2_data[wav_id] = {
            "error_state": error_state,
            "ann": ann,
            "ref": ref
        }
        assert len(ann) == len(ref), wav_id
        for i, j in zip(ann, ref):
            if i!=j:
                error_count += 1
        l2_phoneset += ann
        l2_phoneset += ref
    l2_wav_ids = list(l2_data.keys())

    ################# TIMIT  ################# 
    def finfo(_fname):
        fname = _fname.split('/')
        sa = fname[-1].split('.')[0][:2]
        dtype = fname[-4]
        phones = list(map(lambda x: x.strip().split(' ')[-1], open(_fname, 'r').readlines()))
        wav_id = "-".join(_fname.split('.')[0].split('/')[-4:])
        return dtype, wav_id, sa, phones

    search_path = os.path.join(dir_path, "timit/**/*.phn")
    timit_data = {}
    for fname in glob.iglob(search_path, recursive=True):
        dtype, wav_id, sa, phones = finfo(fname)
        if sa == 'sa':
            continue
        timit_data[wav_id] = phones

    phone39 = []
    with open('dataset/timit/phone39.table', 'r') as f:
        lines = f.readlines()
        for line in lines:
            phone39.append(line.strip().split(' ')[-1])

    ################# dump to disk  ################# 
    def winfo(fname):
        _fname = fname.split('/')
        if _fname[-4] == 'l2':
            dtype = "l2"
            speaker = _fname[-3]
            wav_id = speaker + '-' + _fname[-1].split('.')[0].split('_')[1]
        elif _fname[-5] == 'timit':
            dtype = "timit"
            wav_id = "-".join(fname.split('.')[0].split('/')[-4:])
            speaker = _fname[-4]
        return dtype, wav_id, speaker

    # scan corresponding audio files
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)

    # write down data list
    with open(os.path.join(args.dest, 'train.tsv'), 'w') as train_f, open(
        os.path.join(args.dest, "valid.tsv"), "w") as valid_f, open(
            os.path.join(args.dest, "test.tsv"), "w") as test_f:
        print(dir_path, file=train_f)
        print(dir_path, file=valid_f)
        print(dir_path, file=test_f)

        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            frames = soundfile.info(fname).frames
            if frames >= 320000:
                continue
            dtype, wav_id, speaker = winfo(fname)
            if dtype == 'l2':
                if wav_id not in l2_wav_ids:
                    continue
                if speaker in args.test_set:
                    print(
                        "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=test_f
                    )
                else:
                    print(
                        "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=train_f
                    )
                    if rand.random() > args.valid_percent:
                        pass
                    else:
                        print(
                            "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=valid_f
                        )
            elif dtype == 'timit':
                if wav_id not in timit_data.keys():
                    pass
                else:
                    print(
                        "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=train_f
                    )
    
    # phoneset
    timit_phone_map_f = open(os.path.join(args.root,'timit','phone39.table'), 'r').readlines()
    #l2_phone_f = open(os.path.join(args.root,'l2','phone.table'), 'r').readlines()
    timit_phone_map = {}
    for line in timit_phone_map_f:
        line = line.strip().split(' ')
        timit_phone_map[line[0]] = line[1]
    assert len(set(timit_phone_map.values())) == 41
    def clean_phone(phns):
        _ = []
        for phn in phns:
            if phn in timit_phone_map.keys():
                _.append(timit_phone_map[phn])
            else:
                _.append(phn)
        _ = [i for i in _ if i!='None']
        return _
    # clean
    tmp_a, tmp_b = [],[]
    for _dtype in ["train", 'valid', 'test']:
        lines = open(os.path.join(args.dest, _dtype + '.tsv'), 'r').readlines()
        with open(os.path.join(args.dest, _dtype + '.phn'), 'w') as phn_f, open(
            os.path.join(args.dest, _dtype + '.ref'), 'w') as ref_f:
            for line in lines[1:]:
                fname =  os.path.join(lines[0].strip(), line.split('\t')[0])
                dtype, wav_id, speaker = winfo(fname)
                if dtype == 'l2':
                    ann, ref = l2_data[wav_id]['ann'], l2_data[wav_id]['ref']
                    ann, ref = clean_phone(ann), clean_phone(ref)
                    tmp_a += ann
                    tmp_b += ref
                elif dtype == 'timit':
                    ann = ['sil'] + timit_data[wav_id] +  ['sil']
                    ref = ann
                print(" ".join(ann), file=phn_f)
                print(" ".join(ref), file=ref_f)
    for i in sorted(list(set(tmp_a))):
        print(i)
    
def split_ref_err(content):
    reference, error, raw = [], [], []
    for i in content:
        i = i.split(',')
        if len(i) > 1: # error (s,d,a)
            error.append(1)
            reference.append(i[1].upper())
        else:
            reference.append(i[0].upper())
            error.append(0)
        raw.append(i[0].upper())
    return reference, error, raw


def convert(index_file, write_file, annotation):
    new_errors = {}
    raws = {}
    with open(index_file, 'r') as index_f, open(write_file, 'w') as write_f:
        lines = index_f.readlines()[1:]
        for line in lines:
            wav_id = line.split('/')[-3] + "-" + line.split('/')[-1].split('.')[0].split('_')[-1]
            phns, new_error = get_phn(annotation[wav_id]['reference'], annotation[wav_id]['error'])
            raw, raw_error = get_phn(annotation[wav_id]['raw'], annotation[wav_id]['error'])
            assert len(phns) == len(new_error) == len(raw) == len(raw_error), wav_id

            new_errors[wav_id] = new_error
            raws[wav_id] = raw
            print(" ".join(phns), file=write_f)
    return new_errors, raws

def get_phn(phns):
    phns_wo_number = []
    for phn in phns:
        phn = phn.strip().lower()
        if len(phn) == 0:
            print(phn)
            continue
        if '0' <= phn[-1] <= '9' or phn[-1] == ')' or phn[-1] == '`' or phn[-1] == '_' or phn[-1] == '*':
            phn = phn[:-1]
            if phn[-1] == '`' or phn[-1] == '*' or '0' <= phn[-1] <= '9':
                phn = phn[:-1]
        # if phn in phn_map.keys():
        #     phn = phn_map[phn]
        if phn == '':
            continue
        phns_wo_number.append(phn)
    return phns_wo_number

if __name__ == "__main__":
    phoneset = [] # for test
    parser = get_parser()
    args = parser.parse_args()
    main(args)
