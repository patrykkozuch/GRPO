import numpy as np
from string import punctuation

def four_chars_reward(prompts: list[list[dict[str, str]]], completions: list[list[dict[str, str]]], *args, **kwargs):
    rewards = []

    for completion in completions:
        # Cleanup all punctionations
        content = completion[0]['content']
        answer = content.replace("<think>", "").replace("</think>", "").replace("\n", " ")
        answer = "".join(["" if c in punctuation else c for c in answer])
        words = answer.strip().split()
        rewards.append(
            sum([len(w.strip()) == 4 for w in words]) / len(words)
        )

    return rewards

def has_finished_thinking(prompts: list[list[dict[str, str]]], completions: list[list[dict[str, str]]], *args, **kwargs):
    rewards = []

    for completion in completions:
        # Cleanup all punctionations
        content = completion[0]['content']
        if "</think>" in content:
            rewards.append(1)
        else:
            rewards.append(0)

    return rewards

import re

def preparation_time_reward(prompts, completions, *args, **kwargs):
    rewards = []
    pattern = re.compile(r"<preparation_time>.+</preparation_time>", re.DOTALL)
    for completion in completions:
        content = completion[0]['content']
        rewards.append(1.0 if re.search(pattern, content) else 0.0)
    return rewards

def portions_reward(prompts, completions, *args, **kwargs):
    rewards = []
    pattern = re.compile(r"<portions>.+</portions>", re.DOTALL)
    for completion in completions:
        content = completion[0]['content']
        rewards.append(1.0 if re.search(pattern, content) else 0.0)
    return rewards

def ingredients_structure_reward(prompts, completions, *args, **kwargs):
    rewards = []
    pattern = re.compile(r"<ingredients>(.+)</ingredients>", re.DOTALL)
    ingredient_pattern = re.compile(r"<ingredient>.+</ingredient>")
    for completion in completions:
        content = completion[0]['content']
        match = re.search(pattern, content)
        if match:
            inner = match.group(1)
            ingredients = re.findall(ingredient_pattern, inner)
            rewards.append(1 if len(ingredients) else 0)
        else:
            rewards.append(0.0)
    return rewards

def steps_structure_reward(prompts, completions, *args, **kwargs):
    rewards = []
    pattern = re.compile(r"<steps>(.+)</steps>", re.DOTALL)
    step_pattern = re.compile(r"<step>.+</step>")
    for completion in completions:
        content = completion[0]['content']
        match = re.search(pattern, content)
        if match:
            inner = match.group(1)
            steps = re.findall(step_pattern, inner)
            rewards.append(1 if len(steps) else 0)
        else:
            rewards.append(0.0)
    return rewards

def tag_order_reward(prompts, completions, *args, **kwargs):
    rewards = []
    pattern = re.compile(
        r"<preparation_time>.*?</preparation_time>.*?"
        r"<portions>.*?</portions>.*?"
        r"<ingredients>.*?<ingredient>.*?</ingredient>.*?</ingredients>.*?"
        r"<steps>.*?<step>.*?</step>.*?</steps>",
        re.DOTALL
    )
    for completion in completions:
        content = completion[0]['content']
        rewards.append(1.0 if re.search(pattern, content) else 0.0)
    return rewards

