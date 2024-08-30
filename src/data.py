import json

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.trg = []
        self.label = []
        
        PROMPT = '''You are a helpful AI assistant. Your task is to answer questions based on the given conversation context accurately and kindly. Continuously track and integrate the evolving context of the conversation, referencing previous parts of the conversation when necessary to ensure your responses remain relevant, accurate, and contextually coherent. Provide concise and relevant answers. Handle multilingual contexts appropriately, responding in the language of the question while understanding and appropriately addressing subtle language nuances, idiomatic expressions, and cultural references. Always be polite and empathetic in your responses. If a question is ambiguous, proactively identify potential ambiguities and provide multiple interpretations or request additional information to ensure accurate and relevant answers. Develop a deep semantic understanding to recognize and correctly interpret synonyms, paraphrased expressions, variations in phrasing, tone, and formality, ensuring flexibility and robustness in your responses. Continuously adapt and fine-tune your responses based on the latest information and context-specific nuances. Evaluate responses not only based on exact string matching but also on semantic similarity to ensure alignment with the intended meaning. Regularly analyze your responses to identify and learn from errors, particularly those related to confusing interest with frequency, and continuously refine your understanding and accuracy. Adapt and improve your responses over time through reinforcement learning, using feedback to enhance your accuracy and relevance. Fine-tune your responses with domain-specific data, particularly in contexts where interest and frequency are correlated, to improve your understanding and accuracy in specialized contexts. Incorporate human-in-the-loop mechanisms where human reviewers periodically check and correct your responses, ensuring continuous improvement and high accuracy. Actively seek and incorporate user feedback to continuously refine your responses, adapting to user preferences and improving accuracy and relevance over time. Ensure cultural and linguistic sensitivity by understanding and respecting regional dialects, cultural norms, and context-specific nuances, especially in multilingual contexts. 당신은 유능한 AI 어시스턴트 입니다. 주어진 대화 문맥에 기반하여 질문에 대해 정확하고 친절하게 답변해주세요. 대화의 문맥을 이해한 후에 답변을 제공하세요. 간결하고 관련성 있는 답변을 제공하세요. 다국어 문맥을 적절하게 처리하고, 질문의 언어로 답변하세요. 항상 정중하고 공감하는 태도로 답변하세요. 질문이 모호한 경우, 명확한 답변을 제공하거나 추가 정보를 요청하세요. 유연성과 견고성을 보장하기 위해 동의어와 바꿔 말한 표현을 유효한 답변으로 인식하고 수용하세요. 최신 정보와 문맥별 뉘앙스에 따라 지속적으로 적응하고 답변을 미세 조정하세요. 정확한 문자열 일치뿐만 아니라 의미적 유사성을 기반으로 답변을 평가하여 의도된 의미와의 일치를 보장하세요. 특히 다국어 문맥에서 문화적 및 언어적 민감성을 고려하여 답변하세요.'''
        
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        with open(fname, "r") as f:
            data = json.load(f)
            
        def get_reference_utterances(conversation, reference_ids):
            reference_utterances = []
            for ref_id in reference_ids:
                for cvt in conversation:
                    if cvt['utterance_id'] == ref_id:
                        reference_utterances.append(f"화자{cvt['speaker']}: {cvt['utterance']}")
                        break
            return "\n".join(reference_utterances)
        
        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            reference_utterances = f"[Reference Utterances]\n{get_reference_utterances(inp['conversation'], inp['reference_id'])}\n"

            # 카테고리에 따른 조건문
            question = "[Question]\n대화의 맥락을 바탕으로 다음을 추론하세요:\n"

            if inp['category'] == "전제":
                question += "화자1과 화자2의 발화[Reference Utterances]에서 대화에서 암묵적으로 가정된 사실이나 기본적인 믿음인 전제."
            elif inp['category'] == "원인":
                question += "화자1과 화자2의 발화[Reference Utterances]에서 특정 발화나 행동을 유발한 근본적인 이유."
            elif inp['category'] == "동기":
                question += "화자의 발화[Reference Utterances]를 이끌어낸 감정이나 내면의 욕구인 동기."
            elif inp['category'] == "반응":
                question += "상대 화자의 발화[Reference Utterances]나 행동에 대한 화자의 감정적 또는 행동적 반응."
            else: #후행사건
                question += "[Reference Utterances]를 바탕으로, 이 대화 이후 화자1과 화자2에게 발생할 가능성이 높은 결과나 상황."

            question += "\n중요한 정보를 놓치지 않도록 대화의 주요 맥락과 흐름을 고려하십시오. 추론한 내용을 주어진 3개의 Option과 비교해, 가장 유사한 맥락의 지문을 선택하십시오."
            
            chat = chat + "\n\n" + reference_utterances + "\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat

        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = ""
            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
                
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
