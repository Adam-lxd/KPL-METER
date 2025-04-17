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


def merge_gpt_des_to_slake(slake_path):
    for split in ["train","test","val"]:
        path = f'{slake_path}/vqa_slack_{split}.arrow'
        df = pa.ipc.open_file(path).read_pandas()
        cui_gpt = pd.read_csv("./slake_knowledge.csv")
        big_des = []
        for idx, ky_wrods_id in enumerate(df["key_words_id"]):
            cui_des_ls = []
            for cui_ls_idx, cui_ls in  enumerate(ky_wrods_id):
                des_str = ""
                for idx_cui, cui in enumerate(cui_ls):
                    des = cui_gpt.loc[cui_gpt['id'] == cui, "Description-GPT"].values
                    key = df.loc[idx,"key_words"][cui_ls_idx][idx_cui]
                    if des.size == 0:
                        continue
                    else:
                        des = des[0]
                    if "None" in des: continue
                    
                    if des_str != "":
                        des_str += "</s> "
                
                    des_str += key+" is "+des
                cui_des_ls.append(des_str)



            big_des.append(cui_des_ls)

        df["prompt_text"] = big_des
        new_path = f'{slake_path}/vqa_slack_{split}.arrow'
        table = pa.Table.from_pandas(df)
        with pa.OSFile(new_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

def merge_gpt_des_to_vqa2019(vqa2019_path):
    for split in ["train","test","val"]:
        path = f'{vqa2019_path}/vqa_medvqa_2019_{split}.arrow'
        df = pa.ipc.open_file(path).read_pandas()
        cui_gpt = pd.read_csv("./vqa2019_knowledge.csv")
        big_des = []
        for idx, ky_wrods_id in enumerate(df["key_words_id"]):
            cui_des_ls = []
            for cui_ls_idx, cui_ls in  enumerate(ky_wrods_id):
                des_str = ""
                for idx_cui, cui in enumerate(cui_ls):
                    des = cui_gpt.loc[cui_gpt['id'] == cui, "Description-GPT"].values
                    key = df.loc[idx,"key_words"][cui_ls_idx][idx_cui]
                    if des.size == 0:
                        continue
                    else:
                        des = des[0]
                    if "None" in des: continue
                    
                    if des_str != "":
                        des_str += "</s> "
                
                    des_str += key+" is "+des
                cui_des_ls.append(des_str)



            big_des.append(cui_des_ls)

        df["prompt_text"] = big_des
        new_path = f'{vqa2019_path}/vqa_medvqa_2019_{split}.arrow'
        table = pa.Table.from_pandas(df)
        with pa.OSFile(new_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

merge_knowledge_to_roco(roco_path="./roco_arrow/")
merge_gpt_des_to_slake(slake_path="./slake_arrow/")
merge_gpt_des_to_vqa2019(vqa2019_path="./vqa2019_arrow/")