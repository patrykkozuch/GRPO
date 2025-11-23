from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": ["recipes.txt"]})["train"]


def transform(example):
    parts = example["text"].split("\n\n", 1)
    return {
        "prompt": [
            {
                "content": """You are cook assistant. Based on the user request, provide a recipe. 
                The recipe should have following structure:
                <preparation_time>Preparation time in minutes</preparation_time>
                <portions>Number of portions</portions>
                <ingredients><ingredient>First ingredient</ingredient><ingredient>Another one</ingredient>(all the other ingredients)</ingredients>
                <steps><step>First step to prepare a dish</step><step>Next step</step>(all the other steps)</steps>
                """,
                "role": "system"
            },
            {
                "content": parts[0],
                "role": "user"
            }
        ]
    }


dataset = dataset.map(transform)
dataset = dataset.remove_columns(["text"])

dataset.save_to_disk("dataset")
