from datasets import load_dataset
import re
import random
import os
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import time
from dotenv import load_dotenv


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

ds = ds["test"]
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANSWER_TRIGGER = "The answer is"
INVALID_ANS = "[invalid]"
load_dotenv()

APIKEY = os.getenv("APIKEY")
N_SHOT = 8
COT_FLAG = True


def create_demo_text(n_shot=8, cot_flag=True):
    examples = [
        {
            "question": "Which gas is most abundant in Earth's atmosphere?",
            "choices": {
                "A": "Oxygen",
                "B": "Nitrogen",
                "C": "Carbon Dioxide",
                "D": "Argon",
            },
            "chain": "Earth's atmosphere composition is approximately 78% nitrogen and 21% oxygen. While oxygen is important, nitrogen is clearly the most abundant.",
            "answer": "B",
        },
        {
            "question": "What process do plants use to convert sunlight into energy?",
            "choices": {
                "A": "Respiration",
                "B": "Transpiration",
                "C": "Photosynthesis",
                "D": "Fermentation",
            },
            "chain": "Plants convert sunlight into chemical energy through photosynthesis, which uses chlorophyll to capture light energy.",
            "answer": "C",
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "choices": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
            "chain": "Mars appears reddish due to iron oxide on its surface and is commonly referred to as the Red Planet.",
            "answer": "B",
        },
        {
            "question": "What is the main function of roots in plants?",
            "choices": {
                "A": "Photosynthesis",
                "B": "Absorb water/nutrients",
                "C": "Produce seeds",
                "D": "Attract pollinators",
            },
            "chain": "Roots anchor plants and absorb water/nutrients from soil. Photosynthesis occurs in leaves.",
            "answer": "B",
        },
        {
            "question": "Which animal group does a frog belong to?",
            "choices": {"A": "Mammal", "B": "Reptile", "C": "Amphibian", "D": "Fish"},
            "chain": "Frogs live both in water (as tadpoles) and land (as adults), characteristic of amphibians.",
            "answer": "C",
        },
        {
            "question": "What causes tides on Earth?",
            "choices": {
                "A": "Earth's rotation",
                "B": "Wind patterns",
                "C": "Moon's gravity",
                "D": "Ocean currents",
            },
            "chain": "Tides are primarily caused by the gravitational pull of the moon on Earth's oceans.",
            "answer": "C",
        },
        {
            "question": "Which is a conductor of electricity?",
            "choices": {"A": "Plastic", "B": "Wood", "C": "Rubber", "D": "Copper"},
            "chain": "Metals like copper are good conductors. Plastic, wood, and rubber are insulators.",
            "answer": "D",
        },
        {
            "question": "What is the smallest prime number?",
            "choices": {"A": "1", "B": "2", "C": "3", "D": "5"},
            "chain": "Prime numbers have exactly two distinct factors. 2 is the smallest and only even prime number.",
            "answer": "B",
        },
    ]

    random.shuffle(examples)
    demo_text = ""

    for ex in examples[:n_shot]:
        choices = "\n".join([f"{k}: {v}" for k, v in ex["choices"].items()])
        demo_text += f"Q: {ex['question']}\n{choices}\nA: {ex['chain']} {ANSWER_TRIGGER} {ex['answer']}.\n\n"

    return demo_text


def build_prompt(sample, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    choices = "\n".join(
        [
            f"{k}: {v}"
            for k, v in zip(sample["choices"]["label"], sample["choices"]["text"])
        ]
    )
    return f"{demo}Q: {sample['question']}\n{choices}\nA:"


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    answer_trigger = ANSWER_TRIGGER.lower()

    if answer_trigger in model_pred:
        answer_part = model_pred.split(answer_trigger)[-1].strip()
        match = re.search(r"\b(a|b|c|d)\b", answer_part)
        return match.group(1).upper() if match else INVALID_ANS

    # If no trigger found, search for any capital letter choice
    match = re.search(r"\b(A|B|C|D)\b", model_pred)
    return match.group(0) if match else INVALID_ANS


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    print(f"Model answer: {model_answer}, GT answer: {gt_answer}")
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def ask_question(input_text):
    # prepare prompt for one word answer

    url = "https://api_v2.futurixai.com/api/lara/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": APIKEY,
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert assistant. Analyze the question and options carefully. "
                "Provide step-by-step reasoning to determine the correct answer from the given choices. "
                "Conclude your response with 'The answer is X' where X is the correct option letter.",
            },
            {"role": "user", "content": input_text},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
    }

    response = requests.post(url, headers=headers, json=payload, verify=False)
    response.raise_for_status()
    return response.json()["answer"].strip()


def getIndexOfAnswerLabel(label):
    if label == "A":
        return 0
    elif label == "B":
        return 1
    elif label == "C":
        return 2
    elif label == "D":
        return 3


def process_sample(sample):
    input_text = build_prompt(sample, N_SHOT, COT_FLAG)
    model_completion = ask_question(input_text)
    model_answer = clean_answer(model_completion)
    correct_answer = sample["answerKey"]

    responsejson = {
        "id": sample["id"],
        "question": sample["question"],
        "choices": sample["choices"],
        "correct_answer": correct_answer,
        "model_answer": model_answer,
        "model_completion": model_completion,
        "is_correct": model_answer == correct_answer,
    }

    return model_answer == correct_answer, responsejson


def main():

    answers = []
    complete_response = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_sample, sample) for sample in ds]

        with tqdm(as_completed(futures), total=len(ds)) as pbar:
            for future in pbar:
                is_cor, responsejson = future.result()
                answers.append(is_cor)
                complete_response.append(responsejson)

                # Calculate accuracy
                accuracy = (
                    float(sum(answers)) / len(answers) if len(answers) > 0 else 0.0
                )

                # Update the progress bar's postfix with the current accuracy
                pbar.set_postfix({"Accuracy": f"{accuracy:.2%}"})

                print(
                    f"Num of total question: {len(answers)}, "
                    f"Correct num: {sum(answers)}, "
                    f"Accuracy: {accuracy:.2%}."
                )

    with open(os.path.join("results.txt"), "w") as f:
        for answer in answers:
            print(answer, file=f)

    # write complete responses to a json file
    with open(os.path.join("complete_response.json"), "w") as f:
        json.dump(complete_response, f, indent=4)

    with open(os.path.join("scores.txt"), "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}.",
            file=f,
        )
    while True:
        time.sleep(5)
        print("Completed")


if __name__ == "__main__":
    main()
