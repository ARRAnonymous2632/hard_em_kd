import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

def batch_perplexity(texts, model_name="gpt2", max_tokens=4096):
    """
    Compute perplexity scores for a list of text instances using batching based on a maximum total token count.

    Instead of using a fixed number of texts per batch, this function groups texts together such that
    (number_of_texts_in_batch * max_sequence_length_in_batch) <= max_tokens.
    
    Parameters:
        texts (List[str]): A list of text strings.
        model_name (str): The Hugging Face model name (e.g., 'gpt2').
        max_tokens (int): The maximum total tokens (after padding) to allow per batch.
        
    Returns:
        List[float]: A list of perplexity scores (one per text).
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # For GPT-2, set the pad token to the eos token if not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Use half-precision if possible
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=dtype)
    model.eval()

    perplexities = []

    # Variables to accumulate a batch
    current_batch = []
    current_batch_max_length = 0  # maximum token length among texts in the current batch

    # Iterate over texts to form batches based on max_tokens constraint
    for text in tqdm(texts, desc="Perplexity"):
        # Estimate token length for the text (using truncation and adding special tokens)
        encoding = tokenizer(text, truncation=True, add_special_tokens=True)
        token_length = len(encoding["input_ids"])

        # If the current batch is not empty, determine the new maximum length if we add this text.
        new_batch_max_length = max(current_batch_max_length, token_length) if current_batch else token_length
        new_batch_size = len(current_batch) + 1

        # Estimate total tokens if this text is added (each text will be padded to new_batch_max_length).
        estimated_total_tokens = new_batch_size * new_batch_max_length

        # If adding this text would exceed the allowed max_tokens, process the current batch.
        if current_batch and (estimated_total_tokens > max_tokens):
            perplexities.extend(
                process_batch(current_batch, tokenizer, model, device)
            )
            # Reset the batch.
            current_batch = []
            current_batch_max_length = 0

        # Add the text to the current batch and update the maximum length.
        current_batch.append(text)
        current_batch_max_length = max(current_batch_max_length, token_length)

    # Process any remaining texts in the last batch.
    if current_batch:
        perplexities.extend(
            process_batch(current_batch, tokenizer, model, device)
        )

    return perplexities


def process_batch(batch_texts, tokenizer, model, device):
    """
    Process a batch of texts: tokenize, compute loss per instance, and convert to perplexity.
    
    Parameters:
        batch_texts (List[str]): List of text strings.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face causal language model.
        device: Torch device.
    
    Returns:
        List[float]: Perplexity scores for each text in the batch.
    """
    # Tokenize with padding (and truncation) for the current batch.
    encodings = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Create labels; set positions corresponding to the pad token to -100 so that they are ignored in loss.
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

    # Shift logits and labels so that tokens t are predicted using tokens [0:t-1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute per-token loss (without reduction)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())

    # Create mask to ignore positions with label -100 (padded tokens)
    mask = shift_labels != -100
    # Average loss per instance (divide by number of valid tokens)
    loss_per_instance = (loss * mask.float()).sum(dim=1) / mask.sum(dim=1).float()

    # Convert loss to perplexity
    batch_perplexities = torch.exp(loss_per_instance)

    return batch_perplexities.cpu().tolist()


def main(args):
    # Read input texts from file (one text per line)
    with open(args.input, "r", encoding="utf-8") as f:
        input_data = [line.strip() for line in f.readlines()]
    
    # Compute perplexity scores using the max_tokens batching strategy.
    ppls = batch_perplexity(input_data, model_name=args.model, max_tokens=args.max_tokens)

    # Write perplexity scores to the output file (one score per line)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join([str(ppl) for ppl in ppls]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Path to the input file containing one text per line.")
    parser.add_argument("--output", required=True, help="Path to the output file for perplexity scores.")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum total tokens (after padding) per batch.")
    parser.add_argument("--model", default="gpt2", help="Hugging Face model name (e.g., 'gpt2').")

    args = parser.parse_args()
    main(args)
