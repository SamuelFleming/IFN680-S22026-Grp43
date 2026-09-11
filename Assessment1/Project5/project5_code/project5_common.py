"""
Shared Project 5 implementation for Forward and Reverse notebooks.

The notebooks own experiment orchestration (mode definition, training loop,
inference restore, narrative). This module owns the plumbing that must stay
identical across modes so Task 3 remains a controlled comparison.
"""

from __future__ import annotations

import math
import pickle
import random
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

# ------------------------------------------------------------
# 1. Constants / model configuration
# ------------------------------------------------------------

PAD_TOKEN = "[PAD]"
EOS_TOKEN = "[EOS]"

DEFAULT_MODEL_HYPERPARAMS = {
    "ninp": 128,
    "nhead": 16,
    "nhid": 64,
    "nlayers": 6,
}

DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_NEW_TOKENS = 5
NUMBER_DIGITS = 3
TRAIN_SIZE = 50_000
VAL_SIZE = 10_000
TEST_SIZE = 10_000

SPLITS_PATH = Path("project5_splits.pkl")
TEST_DATA_PATH = Path("project5_test.pkl")


def model_config(ntoken: int) -> dict:
    """Return the tutorial-aligned Transformer config for a vocabulary size."""
    return {"ntoken": ntoken, **DEFAULT_MODEL_HYPERPARAMS}


# ------------------------------------------------------------
# 2. Tokenisation
# ------------------------------------------------------------

class CharacterLevelTokenizer:
    """Character-level tokenizer for the Project 5 arithmetic vocabulary."""

    def __init__(self):
        self.pad_token = PAD_TOKEN
        self.eos_token = EOS_TOKEN
        self.vocab = (
            [str(x) for x in range(10)]
            + ["+", "-", "="]
            + [PAD_TOKEN, EOS_TOKEN]
        )
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        self.ntokens = len(self.vocab)

        # Input prompts may contain spaces for readability; these are removed.
        allowed_chars = "0123456789+-="
        self.pattern = f"[^{re.escape(allowed_chars)}]"

    def clean(self, text: str) -> str:
        return re.sub(self.pattern, "", text)

    def pre_tokenization(self, text: str) -> list[str]:
        return list(text)

    def encode(self, text: str) -> list[int]:
        return [
            self.token_to_id[c]
            for c in self.pre_tokenization(self.clean(text))
        ]

    def decode(self, token_list: Iterable[int]) -> str:
        return "".join(self.id_to_token[int(i)] for i in token_list)


def build_tokenizer() -> CharacterLevelTokenizer:
    return CharacterLevelTokenizer()


# ------------------------------------------------------------
# 3. Arithmetic data
# ------------------------------------------------------------

def convert_datapoint(a_int: int, b_int: int, operation: str) -> tuple[str, str]:
    """Convert two operands and an operation into a (prompt, answer) tuple."""
    if operation == "+":
        result = a_int + b_int
    elif operation == "-":
        result = a_int - b_int
    else:
        raise ValueError("operation must be '+' or '-'")

    return f"{a_int}{operation}{b_int}=", str(result)


def sample_datapoint(
    number_digits: int = NUMBER_DIGITS,
    operation: Optional[str] = None,
) -> tuple[str, str]:
    """Sample one addition or subtraction example with non-negative operands."""
    max_value = (10 ** number_digits) - 1
    a_int = random.randint(0, max_value)
    b_int = random.randint(0, max_value)

    if operation is None:
        operation = random.choice(["+", "-"])

    return convert_datapoint(a_int, b_int, operation)


def generate_unique_operation_samples(
    operation: str,
    n_samples: int,
    seen_prompts: set[str],
    number_digits: int = NUMBER_DIGITS,
) -> list[tuple[str, str]]:
    samples = []
    while len(samples) < n_samples:
        prompt, answer = sample_datapoint(number_digits, operation)
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            samples.append((prompt, answer))
    return samples


def generate_balanced_splits(
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
    number_digits: int = NUMBER_DIGITS,
    seed: Optional[int] = None,
) -> tuple[list, list, list]:
    """
    Generate balanced addition/subtraction train/val/test splits.

    Answers are always stored in normal (Forward) string form.
    Reverse mode applies target reversal later, at batching time.
    """
    assert train_size % 2 == 0 and val_size % 2 == 0 and test_size % 2 == 0

    if seed is not None:
        random.seed(seed)

    seen_prompts: set[str] = set()
    per_operation_total = (train_size + val_size + test_size) // 2

    addition_data = generate_unique_operation_samples(
        "+", per_operation_total, seen_prompts, number_digits
    )
    subtraction_data = generate_unique_operation_samples(
        "-", per_operation_total, seen_prompts, number_digits
    )

    train_half = train_size // 2
    val_half = val_size // 2
    test_half = test_size // 2

    data_train = (
        addition_data[:train_half]
        + subtraction_data[:train_half]
    )
    data_val = (
        addition_data[train_half:train_half + val_half]
        + subtraction_data[train_half:train_half + val_half]
    )
    data_test = (
        addition_data[train_half + val_half:train_half + val_half + test_half]
        + subtraction_data[train_half + val_half:train_half + val_half + test_half]
    )

    random.shuffle(data_train)
    random.shuffle(data_val)
    random.shuffle(data_test)

    return data_train, data_val, data_test


def save_splits(
    data_train: list,
    data_val: list,
    data_test: list,
    splits_path: Path = SPLITS_PATH,
    test_path: Path = TEST_DATA_PATH,
) -> None:
    """Persist all splits, and keep a standalone test pickle for grading compatibility."""
    payload = {
        "train": data_train,
        "validation": data_val,
        "test": data_test,
    }
    with Path(splits_path).open("wb") as f:
        pickle.dump(payload, f)

    with Path(test_path).open("wb") as f:
        pickle.dump(data_test, f)


def load_splits(splits_path: Path = SPLITS_PATH) -> tuple[list, list, list]:
    with Path(splits_path).open("rb") as f:
        payload = pickle.load(f)
    return payload["train"], payload["validation"], payload["test"]


def load_test_set(test_path: Path = TEST_DATA_PATH) -> list:
    with Path(test_path).open("rb") as f:
        return pickle.load(f)


def load_or_create_splits(
    splits_path: Path = SPLITS_PATH,
    test_path: Path = TEST_DATA_PATH,
    seed: int = 680,
) -> tuple[list, list, list]:
    """
    Prefer the shared splits file.

    If only the legacy held-out test pickle exists, first try regenerating all
    splits with ``seed``. When the regenerated test prompts match the saved
    test set, the full regenerated splits are kept (so train/val match the
    original Forward run). Otherwise the saved test set is frozen and train/val
    are rebuilt without overlapping its prompts.
    """
    splits_path = Path(splits_path)
    test_path = Path(test_path)

    if splits_path.exists():
        return load_splits(splits_path)

    data_train, data_val, data_test = generate_balanced_splits(seed=seed)

    if test_path.exists():
        saved_test = load_test_set(test_path)
        saved_prompts = {prompt for prompt, _ in saved_test}
        generated_prompts = {prompt for prompt, _ in data_test}

        if saved_prompts == generated_prompts:
            # Prefer the exact saved test rows (answer strings / order).
            data_test = saved_test
        else:
            test_prompts = saved_prompts
            random.seed(seed)
            seen_prompts = set(test_prompts)
            train_half = TRAIN_SIZE // 2
            val_half = VAL_SIZE // 2

            addition_train_val = generate_unique_operation_samples(
                "+", train_half + val_half, seen_prompts
            )
            subtraction_train_val = generate_unique_operation_samples(
                "-", train_half + val_half, seen_prompts
            )

            data_train = (
                addition_train_val[:train_half]
                + subtraction_train_val[:train_half]
            )
            data_val = (
                addition_train_val[train_half:]
                + subtraction_train_val[train_half:]
            )
            random.shuffle(data_train)
            random.shuffle(data_val)
            data_test = saved_test

    save_splits(data_train, data_val, data_test, splits_path, test_path)
    return data_train, data_val, data_test


def summarise_splits(data_train: list, data_val: list, data_test: list) -> None:
    print(f"Train: {len(data_train):,}")
    print(f"Validation: {len(data_val):,}")
    print(f"Test: {len(data_test):,}")

    train_prompts = {p for p, _ in data_train}
    val_prompts = {p for p, _ in data_val}
    test_prompts = {p for p, _ in data_test}

    print("Train/val overlap:", len(train_prompts & val_prompts))
    print("Train/test overlap:", len(train_prompts & test_prompts))
    print("Val/test overlap:", len(val_prompts & test_prompts))

    for split_name, split_data in [
        ("train", data_train),
        ("val", data_val),
        ("test", data_test),
    ]:
        n_add = sum("+" in prompt for prompt, _ in split_data)
        n_sub = sum("-" in prompt for prompt, _ in split_data)
        print(f"{split_name:<5}: addition={n_add:,}, subtraction={n_sub:,}")


# ------------------------------------------------------------
# 4. Target representations
# ------------------------------------------------------------

TargetTransform = Callable[[str], str]


def forward_target(answer: str) -> str:
    """Identity transform used by Forward mode."""
    return answer


def reverse_target(answer: str) -> str:
    """Reverse the complete answer string, including a leading '-' if present."""
    return answer[::-1]


# ------------------------------------------------------------
# 5. Arithmetic descriptors
# ------------------------------------------------------------

def parse_prompt(prompt: str) -> tuple[int, str, int]:
    """Return (a, operation, b) from a prompt such as '123-45=' or '12+7='."""
    expression = prompt.rstrip("=")
    if "+" in expression:
        a, b = expression.split("+")
        return int(a), "+", int(b)
    if "-" in expression:
        a, b = expression.split("-")
        return int(a), "-", int(b)
    raise ValueError(f"Unsupported prompt: {prompt}")


def count_carries(prompt: str) -> int:
    """Count columns that generate a carry for an addition prompt."""
    a, operation, b = parse_prompt(prompt)
    if operation != "+":
        return 0

    carry = 0
    count = 0
    max_digits = max(len(str(a)), len(str(b)))

    for position in range(max_digits):
        digit_a = (a // (10 ** position)) % 10
        digit_b = (b // (10 ** position)) % 10
        if digit_a + digit_b + carry >= 10:
            count += 1
            carry = 1
        else:
            carry = 0
    return count


def count_borrows(prompt: str) -> int:
    """Count columns that require a borrow for a subtraction prompt."""
    a, operation, b = parse_prompt(prompt)
    if operation != "-":
        return 0

    borrow = 0
    count = 0
    max_digits = max(len(str(a)), len(str(b)))

    for position in range(max_digits):
        digit_a = (a // (10 ** position)) % 10
        digit_b = (b // (10 ** position)) % 10
        adjusted_a = digit_a - borrow

        if adjusted_a < digit_b:
            count += 1
            borrow = 1
        else:
            borrow = 0
    return count


# ------------------------------------------------------------
# 6. Transformer architecture
# ------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used in the Week 7 tutorial."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, d_model, 2).float() / d_model) * (-math.log(1e4))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return x


class CustomEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer that also stores per-head attention weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.attn_weights = attn_weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = CustomEncoderLayer(ninp, nhead, nhid, 0.1)
        self.encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, ntoken)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def forward(self, src):
        mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output_enc = self.encoder(src, mask=mask)
        output_dec = self.decoder(output_enc)
        attention_maps = [layer.attn_weights for layer in self.encoder.layers]
        return F.log_softmax(output_dec, dim=-1), output_enc, attention_maps


# ------------------------------------------------------------
# 7. Batch preparation
# ------------------------------------------------------------

def pad(
    token_list: list[list[int]],
    tokenizer: CharacterLevelTokenizer,
    type_list: str = "prompts",
) -> tuple[list[list[int]], int]:
    """Pad prompt or answer token sequences to a common batch length."""
    assert type_list in ["prompts", "answers"]

    max_length = max(len(x) for x in token_list)
    out = []
    pad_id = tokenizer.token_to_id[PAD_TOKEN]
    eos_id = tokenizer.token_to_id[EOS_TOKEN]

    for x in token_list:
        if type_list == "prompts":
            # Left-pad prompts so '=' aligns across the batch.
            out.append([pad_id] * (max_length - len(x)) + x)
        else:
            # Append EOS, then right-pad answers.
            out.append(
                x
                + [eos_id]
                + [pad_id] * (max_length - len(x))
            )

    return out, max_length


def get_batch(
    data: list[tuple[str, str]],
    start_index: int,
    batch_size: int,
    tokenizer: CharacterLevelTokenizer,
    target_transform: TargetTransform = forward_target,
):
    """
    Return one padded prompt/answer batch.

    Arithmetic prompts remain unchanged. Target answer strings are passed
    through ``target_transform`` before tokenisation (identity for Forward,
    full-string reverse for Reverse).
    """
    batch = data[start_index:start_index + batch_size]

    prompts = [tokenizer.encode(prompt) for prompt, _ in batch]
    answers = [
        tokenizer.encode(target_transform(answer))
        for _, answer in batch
    ]

    padded_prompts, length_prompts = pad(prompts, tokenizer, "prompts")
    padded_answers, length_answers = pad(answers, tokenizer, "answers")

    X = torch.stack(
        [torch.tensor(x, dtype=torch.long) for x in padded_prompts],
        dim=1,
    )
    Y = torch.stack(
        [torch.tensor(y, dtype=torch.long) for y in padded_answers],
        dim=1,
    )

    return X, Y, length_prompts, length_answers


# ------------------------------------------------------------
# 8. Generation
# ------------------------------------------------------------

def generate(model, prompts, new_tokens: int = 5, device=None):
    """Autoregressively append predicted tokens to a batch of prompts."""
    if device is None:
        device = next(model.parameters()).device

    input_tensor = prompts.to(device)

    for _ in range(new_tokens):
        output, _, _ = model(input_tensor)
        last_output = output[-1, :, :]
        token = torch.argmax(last_output, dim=-1).view(1, -1)
        input_tensor = torch.cat((input_tensor, token), dim=0)

    return input_tensor


# ------------------------------------------------------------
# 9. Shared evaluation helpers
# ------------------------------------------------------------

def evaluate_autoregressive(
    model,
    input_data,
    tokenizer: CharacterLevelTokenizer,
    device,
    target_transform: TargetTransform = forward_target,
    batch_size: int = DEFAULT_BATCH_SIZE,
    desc: str = "Validation",
):
    """
    Evaluate token-level and complete-sequence autoregressive accuracy.

    Targets are compared in the transformed space used during training
    (normal order for Forward, reversed order for Reverse).
    """
    model.eval()

    correct_token = 0
    total_token = 0
    correct_sequence = 0
    total_sequence = 0

    batch_starts = range(0, len(input_data), batch_size)
    progress = tqdm(
        batch_starts,
        total=math.ceil(len(input_data) / batch_size),
        desc=desc,
        leave=False,
    )

    pad_id = tokenizer.token_to_id[PAD_TOKEN]

    with torch.no_grad():
        for i in progress:
            prompts, target_answers, length_prompts, length_answers = get_batch(
                input_data,
                i,
                batch_size,
                tokenizer,
                target_transform,
            )

            prompts = prompts.to(device)
            target_answers = target_answers.to(device)

            output = generate(model, prompts, length_answers + 1, device=device)
            answer_tokens = output[length_prompts:, :]

            target_mask = target_answers != pad_id
            equality = answer_tokens == target_answers

            correct_token += torch.logical_and(target_mask, equality).sum().item()
            total_token += target_mask.sum().item()

            correct_sequence += torch.all(
                torch.logical_or(~target_mask, equality),
                dim=0,
            ).sum().item()
            total_sequence += target_answers.shape[1]

            progress.set_postfix(
                token_acc=f"{correct_token / max(total_token, 1):.3f}",
                seq_acc=f"{correct_sequence / max(total_sequence, 1):.3f}",
            )

    return correct_token / total_token, correct_sequence / total_sequence


def decode_generated_answer(
    token_ids,
    tokenizer: CharacterLevelTokenizer,
    restore_prediction: TargetTransform = forward_target,
):
    """
    Decode generated answer tokens up to EOS.

    ``restore_prediction`` maps model output order back to normal numeric text
    (identity for Forward; reverse for Reverse).
    """
    eos_id = tokenizer.token_to_id[EOS_TOKEN]
    pad_id = tokenizer.token_to_id[PAD_TOKEN]

    answer_ids = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == eos_id:
            break
        if token_id == pad_id:
            continue
        answer_ids.append(token_id)

    raw_text = tokenizer.decode(answer_ids)
    restored_text = restore_prediction(raw_text)

    try:
        numeric_value = int(restored_text)
    except ValueError:
        numeric_value = None

    return numeric_value, restored_text, raw_text


def predict_sample(
    model,
    prompt: str,
    tokenizer: CharacterLevelTokenizer,
    restore_prediction: TargetTransform = forward_target,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device=None,
):
    """Generate and decode one arithmetic answer."""
    if device is None:
        device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).view(-1, 1)

    generated = generate(model, prompt_tensor, max_new_tokens, device=device)
    generated_ids = generated[:, 0].detach().cpu().tolist()
    answer_token_ids = generated_ids[len(prompt_ids):]

    predicted, restored_text, raw_text = decode_generated_answer(
        answer_token_ids,
        tokenizer,
        restore_prediction,
    )
    return predicted, restored_text, raw_text


def predict_dataset(
    model,
    data,
    tokenizer: CharacterLevelTokenizer,
    restore_prediction: TargetTransform = forward_target,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device=None,
):
    """Return one prediction record per example using batched generation."""
    if device is None:
        device = next(model.parameters()).device

    records = []
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, len(data), batch_size), desc="Predict"):
            batch = data[start:start + batch_size]
            prompts_encoded = [tokenizer.encode(prompt) for prompt, _ in batch]
            padded_prompts, prompt_length = pad(
                prompts_encoded, tokenizer, "prompts"
            )
            prompt_tensor = torch.stack(
                [torch.tensor(x, dtype=torch.long) for x in padded_prompts],
                dim=1,
            )

            output = generate(
                model, prompt_tensor, max_new_tokens, device=device
            ).detach().cpu()
            generated = output[prompt_length:, :]

            for column, (prompt, answer) in enumerate(batch):
                predicted, restored_text, raw_prediction = decode_generated_answer(
                    generated[:, column].tolist(),
                    tokenizer,
                    restore_prediction,
                )
                actual = int(answer)
                _, operation, _ = parse_prompt(prompt)

                records.append({
                    "prompt": prompt,
                    "operation": operation,
                    "actual": actual,
                    "predicted": predicted,
                    "raw_prediction": raw_prediction,
                    "restored_prediction": restored_text,
                    "exact": predicted == actual,
                    "carry_count": count_carries(prompt),
                    "borrow_count": count_borrows(prompt),
                })

    return records


def accuracy(records: list[dict]) -> float:
    if not records:
        return float("nan")
    return sum(record["exact"] for record in records) / len(records)


def subset(records: list[dict], predicate) -> list[dict]:
    return [record for record in records if predicate(record)]


DIGIT_POSITIONS = {
    "Units": 0,
    "Tens": 1,
    "Hundreds": 2,
    "Thousands": 3,
}


def get_digit_if_present(value, position: int):
    """Return the digit at a place-value position only if it exists."""
    magnitude = str(abs(int(value)))
    if position >= len(magnitude):
        return None
    return int(magnitude[-(position + 1)])


def digit_position_accuracy(records: list[dict], position: int):
    """Accuracy at one digit position for examples that contain that position."""
    correct = 0
    total = 0

    for record in records:
        actual = record["actual"]
        predicted = record["predicted"]
        actual_digit = get_digit_if_present(actual, position)

        if actual_digit is None:
            continue

        total += 1
        if predicted is None:
            continue

        predicted_digit = get_digit_if_present(predicted, position)
        if predicted_digit == actual_digit:
            correct += 1

    if total == 0:
        return None, 0

    return correct / total, total
