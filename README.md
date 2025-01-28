# Shivaay LLM - ARC-Challenge Benchmark Evaluation

This repository contains the evaluation code and results for the **ARC-Challenge Benchmark** using the **Shivaay LLM** with **8-shot testing** and **Chain-of-Thought (CoT)** reasoning. The benchmark measures the model's ability to answer complex science questions.

## Metrics

- **Evaluation Method**: 8-Shot Testing with Chain-of-Thought Reasoning
- **Primary Metric**: Accuracy (%) on the ARC-Challenge dataset

## Prerequisites

- Python 3.6 or higher
- A valid **FuturixAI API Key** (obtain from [Shivaay Playground](https://shivaay.futurixai.com/playground))
- Required packages: `requests`, `python-dotenv`

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/[YourUsername]/shivaay-arc-benchmark.git
   cd shivaay-arc-benchmark
   ```

2. **Set Up a Virtual Environment (recommended):**:

   ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/MacOS
    # OR
    venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies:**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Add API Key:**:

- Create a .env file in the root directory
- Add your FuturixAI API key:
  ```bash
  FUTURIXAI_API_KEY="your_api_key_here"
  ```

---

## Running the Evaluation

Execute the benchmark script:

```bash
python evaluate.py
```

### Output Files

1. `complete_responses.json`:
   Contains detailed model responses in the following format:

```json
[
  {
    "id": "Mercury_SC_408547",
    "question": "The end result in the process of photosynthesis...",
    "choices": {
      "text": ["Chemical energy...", "Light energy...", ...],
      "label": ["A", "B", "C", "D"]
    },
    "correct_answer": "C",
    "model_answer": "C",
    "model_completion": "The beginning of photosynthesis is marked by...",
    "is_correct": true
  }
]
```

2. `scores.txt`:
   Provides final accuracy metrics:

```txt
Num of total questions: 1172, Correct num: 1067, Accuracy: 0.91040955631
```

Final Score: `Accuracy * 100` (e.g., 91.04% in our run).

---

### Notes

- **Invalid Responses**: 23 responses in the test set were tagged [invalid] due to parsing errors. These require manual evaluation.
- **Reproducibility**: To replicate our results, ensure you use the same 8-shot examples provided in the repository.
