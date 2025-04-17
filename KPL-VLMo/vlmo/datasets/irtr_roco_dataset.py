from .base_dataset import BaseDataset
import random
class IRTRROCODataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["irtr_roco_entity_train"]
        elif split == "val":
            names = ["irtr_roco_entity_val"]
        elif split == "test":
            names = ["irtr_roco_entity_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def get_false_prompt_text(self, rep, false_text_first_index, false_text_second_index):
        text = self.all_prompt_texts[false_text_first_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        return {f"false_prompt_text_{rep}": (text, encoding)}
    
    def get_false_text(self, rep, selected_index=None):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        
        return {f"false_text_{rep}": (text, encoding), f"false_index": index, f"false_cap_index": caption_index}
    
    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret["split"]=self.split
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    prompt_text = self.table["prompt_text"][index].as_py()
                    prompt_text = self.get_prompt_text(prompt_text)["text"]
                    ret.update({"prompt_text": prompt_text})
                    ret.update(txt)
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i, selected_index=index))

                for i in range(self.draw_false_text):
                    false_text_dict = self.get_false_text(i, selected_index=index)
                    false_text_first_index = false_text_dict["false_index"]
                    false_text_second_index = false_text_dict["false_cap_index"]
                    del false_text_dict["false_index"]
                    del false_text_dict["false_cap_index"]
                    ret.update(false_text_dict)
                    false_prompt_text = self.get_false_prompt_text(i, false_text_first_index=false_text_first_index, false_text_second_index=false_text_second_index)
                    ret.update(false_prompt_text)
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret
    def __getitem__(self, index):
        return self.get_suite(index)
