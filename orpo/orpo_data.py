import json
from torch.utils.data import Dataset 
from datasets import Dataset as HFDataset  

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        self.data = []
        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

        with open(fname, "r") as f:
            data = json.load(f)

        for example in data:
            chat = [{"role": "system", "content": PROMPT}]
            for cvt in example["input"]['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append({"content": f"화자{speaker}: {utterance}", "role": "user",})
                #chat.append({"role": "user", "content": f"화자{speaker}: {utterance}"})

            question = f"[Question]\n위 대화의 {example['input']['category']}"
            if (ord(example['input']['category'][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question += "로"
            question += " 올바른 지문은?"

            options = f"A. {example['input']['inference_1']}\nB. {example['input']['inference_2']}\nC. {example['input']['inference_3']}"
            chat.append({"role": "user", "content": f"{question}\n\n[Option]\n{options}"})

            chosen = example['input'][example['output']]
            rejected = ", ".join([example['input'][inf] for inf in ["inference_1", "inference_2", "inference_3"] if inf != example['output']])

            self.data.append({
                "prompt": chat,
                "chosen": chosen,
                "rejected": rejected
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #return self.data[idx]
        item = self.data[idx]
        if item["prompt"] is None or item["chosen"] is None or item["rejected"] is None:
            raise ValueError(f"Item at index {idx} contains None values.")
        return item

def custom_to_hf_dataset(custom_dataset):
    data_dict = {"prompt": [], "chosen": [], "rejected": []}
    for item in custom_dataset:
        prompt_text = " ".join([msg["content"] for msg in item["prompt"] if msg["role"] == "user"])
        if prompt_text is None:
            raise ValueError("Prompt text is None.")
        #prompt list 형식 안된다고 해서 string으로 받도록 처리
        #prompt_text = " ".join([msg["content"] for msg in item["prompt"] if msg["role"] == "user"])
        #data_dict["prompt"].append(item["prompt"])
        data_dict["prompt"].append(prompt_text)
        data_dict["chosen"].append(item["chosen"])
        data_dict["rejected"].append(item["rejected"])
    
    return HFDataset.from_dict(data_dict) 
