import pyarrow as pa
import pandas as pd
def merge_knowledge_to_roco(roco_path):
    cui_gpt = pd.read_csv("./roco_100%_knowledge.csv")
    for split in ["train", "val", "test"]:
        path = f'{roco_path}/irtr_roco_{split}.arrow'
        big_des = []
        df = pa.ipc.open_file(path).read_pandas()
        for idx, ky_wrods_id in enumerate(df["key_words_id"]):
            des = df.loc[idx, "prompt_text"]
            des_str = ""
            for idx_cui, cui in enumerate(ky_wrods_id):
                des = cui_gpt.loc[cui_gpt['CUI'] == cui, 'Description-GPT'].values
                key = df.loc[idx,"key_words"][idx_cui]
                if "None" in des or des.size == 0:
                    continue
                if des_str != "":
                    des_str += "</s> "
                try:
                    des_str += key+" is "+des
                except:
                    print("skip error cui", cui)
            if des_str == "":
                big_des.append(des_str)
            else:
                big_des.append(des_str[0])
        df["prompt_text"] = big_des
        new_path = f'{roco_path}/irtr_roco_{split}.arrow'
        table = pa.Table.from_pandas(df)
        with pa.OSFile(new_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
merge_knowledge_to_roco(roco_path="./roco_arrow/")