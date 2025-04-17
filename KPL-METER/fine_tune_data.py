from tqdm import tqdm
import spacy
import json
import os
import re
import random
import pandas as pd
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from glossary import normalize_word
import hashlib
from io import BytesIO

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_scibert")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

def parse_a_text(text):
    entities = []
    doc = nlp(text)
    for ent in doc.ents:
        # Noise Filtering
        if len(ent.text) == 1:
            continue
        # Link to UMLS
        if len(ent._.kb_ents) == 0:
            continue
        start_id = ent.start_char
        end_id = ent.end_char
        cuis = ent._.kb_ents
        cuis = [cui[0] for cui in cuis if cui[1] >= 0.95]
        if len(cuis) == 0:
            continue
        entities.append((start_id, end_id, ent.text, cuis[0]))
    return entities
def extract_umls(data, text_key="texts"):
    for split in ["train", "val", "test"]:
        split_data = data[split]
        for sample_idx, sample in tqdm(enumerate(split_data)):
            image_entites = []
            text_entities = []
            text = sample[text_key]
            entities = parse_a_text(text)
            text_entities.append(entities)
            image_entites.extend(entities)
            promptText = "" 
            if sample_idx < 5:
                print(text_entities)
            sample["key_words"] = [ ent[2] for ent in entities ]
            sample["key_words_id"] = [ ent[-1] for ent in entities ]
            sample["prompt_text"] = promptText
    return data
def get_score(occurences):
    return 1.0
def path2rest_vqa(path, split, annotations, label2ans):
    with open(path, "rb") as fp:
        binary = fp.read()
    iid = path
    _annotation = annotations[split][iid]
    _annotation = list(_annotation.items())
    qids, qas = [a[0] for a in _annotation], [a[1] for a in _annotation]
    questions = [qa[0] for qa in qas]
    key_words = [qa[1] for qa in qas]
    key_words_ids = [qa[2] for qa in qas]
    prompt_text = []
    for qa in qas: 
        if len(qa[5]) == 0:
            prompt_text.append("")
        else:
            prompt_text.append(qa[5][0])
    answers = [qa[6] for qa in qas]
    answer_labels = [a["labels"] for a in answers]
    answer_scores = [a["scores"] for a in answers]
    question_types = [a["answer_type"] for a in answers]
    answers = [[label2ans[l] for l in al] for al in answer_labels]
    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, question_types,
            key_words, key_words_ids, prompt_text, split]
def path2rest(path, iid2captions, iid2prompt_kywords, iid2prompt_kyids, iid2prompt_text, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    prompt_kywords = iid2prompt_kywords[name]
    prompt_kyids = iid2prompt_kyids[name]
    prompt_text = iid2prompt_text[name]
    split = iid2split[name]
    return [binary, captions, name, prompt_kywords, prompt_kyids, prompt_text, split]
def make_arrow_vqa(data, dataset_name, save_dir):
    questions_train, questions_val, questions_test = data["train"], data["val"], data["test"]

    # Record Questions
    annotations = dict()
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        _annotation = defaultdict(dict)
        for q in tqdm(questions):
            _annotation[q["img_path"]][q["qid"]] = [q["question"], q["key_words"], q["key_words_id"], q["prompt_text"]]
        annotations[split] = _annotation

    # Construct Vocabulary
    all_major_answers = list()
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        for q in tqdm(questions):
            all_major_answers.append(str(q["answer"]).lower())
    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())
    print("Label size ({}): {}.".format(dataset_name, len(ans2label)))

    # Record Answers
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        _annotation = annotations[split]
        for q in tqdm(questions):
            answers = normalize_word(str(q["answer"]).lower())
            answer_count = {}
            answer_count[answers] = answer_count.get(answers, 0) + 1
            labels = []
            scores = []
            for answer in answer_count:
                assert answer in ans2label
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)
            assert q['answer_type'].strip().lower() == "closed" or q['answer_type'].strip().lower() == "open"
            answer_type = 0 if q['answer_type'].strip().lower() == "closed" else 1
            _annotation[q["img_path"]][q["qid"]].append(
                {"labels": labels, "scores": scores, "answer_type": answer_type})

    # Write to the files
    for split in ["train", "val", "test"]:
        annot = annotations[split]
        annot_paths = [path for path in annot if os.path.exists(path)]
        assert len(annot_paths) == len(annot) or len(annot_paths) == len(annot) - 1
        print("{} set: {} images, {} questions".format(split,
                                                       len(annot),
                                                       len([vv for k, v in annot.items() for kk, vv in v.items()])))

        bs = [
            path2rest_vqa(path, split, annotations, label2ans) for path in tqdm(annot_paths)
        ]
        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "answer_type",
                "key_words",
                "key_words_id",
                "prompt_text",
                "split",
            ],
        )
        table = pa.Table.from_pandas(dataframe)

        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

def make_arrow(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)

    iid2prompt_kywords = defaultdict(list)
    iid2prompt_kyids = defaultdict(list)
    iid2prompt_text = defaultdict(list)

    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])

            iid2prompt_kywords[sample["img_path"]].extend(sample["key_words"])
            iid2prompt_kyids[sample["img_path"]].extend(sample["key_words_id"])
            iid2prompt_text[sample["img_path"]].extend(sample["prompt_text"])
            if sample["img_path"] in iid2split.keys():
                print(sample["img_path"], "exists")
            iid2split[sample["img_path"]] = split
    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    bs = [path2rest(path, iid2captions, iid2prompt_kywords, iid2prompt_kyids, iid2prompt_text, iid2split) 
    for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "key_words",
            "key_words_id", "prompt_text","split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

def prepro_vqa_slack():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "SLAKE-DATAROOT"
    image_root = f"{data_root}/imgs"

    for split, file in zip(["train", "val", "test"], ["train.json", "validate.json", "test.json"]):
        with open(f"{data_root}/{file}", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                if sample["q_lang"] != "en":
                    continue
                img_path = os.path.join(image_root, sample["img_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    extract_umls(data, text_key="question")
    make_arrow_vqa(data, "vqa_slack", "ARROW_PATH")

def prepro_vqa_medvqa2019():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "VQA_2019_DATA_ROOT"
    image_root = "VQA_2019_DATA_ROOT/{}/images"

    offset = 0
    for split in ["train", "val", "test"]:
        samples = open(f"{data_root}/{split}/QA/Modality.csv").read().strip().split("\n") + \
                  open(f"{data_root}/{split}/QA/Organ.csv").read().strip().split("\n") + \
                  open(f"{data_root}/{split}/QA/Plane.csv").read().strip().split("\n")
        samples = [[idx + offset] + question.split("|") for idx, question in enumerate(samples)]
        offset += len(samples)
        for sample in samples:
            img_path = os.path.join(image_root.format(split), sample[1] + ".jpg")
            qid = sample[0]
            question = sample[2]
            answer = sample[3]
            answer_type = "OPEN"
            data[split].append({
                "img_path": img_path,
                "qid": qid,
                "question": question,
                "answer": answer,
                "answer_type": answer_type
            })

    extract_umls(data, text_key="question")
    make_arrow_vqa(data, "vqa_medvqa_2019", "ARROW_PATH")

def prepro_irtr_roco(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    roco_data_root = "ROCO_DATA_ROOT"
    roco_image_root = "ROCO_DATA_ROOT/data/{}/radiology/images/"

    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/captions.txt", "r") as fp:
            lines = fp.read().strip().split("\n")
            random.shuffle(lines)
            for line_idx, line in enumerate(lines):
                str_splits = line.strip().split('\t')
                if len(str_splits) == 2:
                    img_path = os.path.join(roco_image_root.format(split), str_splits[0] + ".jpg")
                    texts = [str_splits[1]]
                    texts = [re.sub(r"\s+", " ", text) for text in texts]
                    texts = [text for text in texts if len(text.split()) >= min_length]
                    if len(texts) > 0:
                        data[split].append({
                            "img_path": img_path,
                            "texts": texts
                        })
    extract_umls(data, text_key="texts")
    make_arrow(data, "irtr_roco", "ARROW_PATH")

if __name__ == '__main__':
    prepro_vqa_slack()
    prepro_vqa_medvqa2019()
    prepro_irtr_roco()