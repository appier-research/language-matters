import json
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, ModernBertForTokenClassification, Qwen2ForTokenClassification, AutoModelForTokenClassification
from typing import List, Optional, Union
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# For older NVIDIA GPUs, suppress Triton compiler error and fall back to eager execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

# Disable Triton compiler warnings
import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def predict_step_splits(
        text: str,
        model: ModernBertForTokenClassification,
        tokenizer: AutoTokenizer,
        sep_token: str = "[SEP]",
        device: Optional[torch.device] = None
    ) -> List[int]:
    """
    Predict whether each [SEP] token should split into a new step.

    Args:
        text: Input text with [SEP] tokens
        model: Trained ModernBertForTokenClassification model
        tokenizer: Tokenizer for the model
        sep_token: Token used to separate text segments
        device: Device to run inference on (defaults to CUDA if available)

    Returns:
        List of binary predictions (0 or 1) for each [SEP] token
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

    # Find SEP tokens
    input_ids = inputs["input_ids"][0]
    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

    # Get prediction for each SEP token
    sep_predictions = [predictions[0, pos].item() for pos in sep_positions]
    return sep_predictions

def format_steps(text: str, split_predictions: List[int], sep_token: str = "[SEP]") -> List[str]:
    """
    Format text into steps based on the split predictions.

    Args:
        text: Input text with [SEP] tokens
        split_predictions: Binary predictions for each [SEP] token
        sep_token: Token used to separate text segments

    Returns:
        List of formatted steps
    """
    text_parts = text.split(sep_token)
    formatted_steps = []
    current_step = text_parts[0]

    for i, pred in enumerate(split_predictions):
        if i + 1 < len(text_parts):
            if pred == 1:  # This SEP should be a step break
                formatted_steps.append(current_step.strip())
                current_step = text_parts[i + 1]
            else:  # This SEP should not be a step break
                current_step += " " + text_parts[i + 1]+'\n'

    # Add the last step
    if current_step.strip():
        formatted_steps.append(current_step.strip())

    return formatted_steps

def extract_reasoning_from_output(
    output: str,
    end_of_thinking_token: str = "</think>",
) -> str:
    """
    Extract the reasoning from the output string.
    """
    thinking_end_pos = output.find(end_of_thinking_token)
    if thinking_end_pos == -1:
        reasoning = output
    else:
        reasoning = output[:thinking_end_pos]
    return reasoning

def main():
    parser = argparse.ArgumentParser(description="Use a trained model to predict step splits")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        help="Path to a jsonl file with each row containing LLM reasoning output to process",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        help="Path to save the output steps",
    )
    parser.add_argument(
        "--sep_token",
        type=str,
        default="[SEP]",
        help="Token used to separate text segments",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    args = parser.parse_args()

    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    # Count the number of lines in input_jsonl
    line_count = 0
    with open(args.input_jsonl, 'r') as f:
        for _ in f:
            line_count += 1
    logger.info(f"Processing {line_count} examples from {args.input_jsonl}")

    logger.info("Running inference...")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text('')  # clear the file
    with open(args.input_jsonl, 'r') as f:
        for line in tqdm(f, total=line_count, dynamic_ncols=True):
            data = json.loads(line)
            text = extract_reasoning_from_output(data["output"])
            # Get predictions
            text = text.replace('\n\n', args.sep_token).replace('\n', args.sep_token)
            split_predictions = predict_step_splits(text, model, tokenizer, args.sep_token)

            # Format steps
            formatted_steps = format_steps(text, split_predictions, args.sep_token)

            with open(args.output_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "input": data["input"],
                    "reasoning": text,
                    "formatted_steps": formatted_steps,
                    "num_steps": len(formatted_steps),
                }) + '\n')

            if args.debug:
                break

if __name__ == "__main__":
    main()
