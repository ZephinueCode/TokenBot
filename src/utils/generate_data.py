# src/utils/generate_data.py

import json
import random
import os
from .parameters import HYPERPARAMS as HP

# ================= CONFIGURATION =================
OUTPUT_FILE = HP.SFT_DATA_PATH
SAMPLES_PER_ACTION = HP.SFT_SAMPLES_PER_ACTION # e.g., 50
SAMPLES_PER_EXPLANATION = int(HP.SFT_SAMPLES_PER_ACTION / 6)

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ================= VOCABULARY (Kept same as yours, simplified for brevity) =================
VOCAB = {
    "ui_element": ["button", "icon", "link", "menu", "field", "card", "search bar"],
    "container": ["page", "screen", "window", "list", "panel"],
    "move_verb": ["Move", "Shift", "Position the cursor", "Slide"],
    "click_verb": ["Click", "Tap", "Select", "Hit"],
    "far_desc": ["significantly", "a lot", "far", "by 300 pixels"],
    "mid_desc": ["moderately", "a bit", "by 100 pixels"],
    "clo_desc": ["slightly", "a tiny bit", "pixel-perfect adjustment"],
    "up": ["up", "upwards"], "down": ["down", "downwards"],
    "left": ["left", "leftwards"], "right": ["right", "rightwards"],
    "text_content": ["Hello world", "search query", "user@example.com"],
    "type_verb": ["Type", "Enter", "Input"],
    "end_action": ["Stop task", "Finish interaction", "Done"]
}

# ================= TOKEN META (Same as yours) =================
TOKEN_META = {
    "<MOVE_UP_FAR>":    {"type": "move", "dir": "up", "dist": "far"},
    "<MOVE_UP_MID>":    {"type": "move", "dir": "up", "dist": "mid"},
    "<MOVE_UP_CLO>":    {"type": "move", "dir": "up", "dist": "clo"},
    "<MOVE_DOWN_FAR>":  {"type": "move", "dir": "down", "dist": "far"},
    "<MOVE_DOWN_MID>":  {"type": "move", "dir": "down", "dist": "mid"},
    "<MOVE_DOWN_CLO>":  {"type": "move", "dir": "down", "dist": "clo"},
    "<MOVE_LEFT_FAR>":  {"type": "move", "dir": "left", "dist": "far"},
    "<MOVE_LEFT_MID>":  {"type": "move", "dir": "left", "dist": "mid"},
    "<MOVE_LEFT_CLO>":  {"type": "move", "dir": "left", "dist": "clo"},
    "<MOVE_RIGHT_FAR>": {"type": "move", "dir": "right", "dist": "far"},
    "<MOVE_RIGHT_MID>": {"type": "move", "dir": "right", "dist": "mid"},
    "<MOVE_RIGHT_CLO>": {"type": "move", "dir": "right", "dist": "clo"},
    "<CLICK_SHORT>":    {"type": "click", "mode": "short"},
    "<CLICK_LONG>":     {"type": "click", "mode": "long"},
    "<GO_BACK>":        {"type": "nav", "mode": "back"},
    "<GO_HOME>":        {"type": "nav", "mode": "home"},
    "<SCROLL_UP>":      {"type": "scroll", "dir": "up"},
    "<SCROLL_DOWN>":    {"type": "scroll", "dir": "down"},
    "<SCROLL_LEFT>":    {"type": "scroll", "dir": "left"},
    "<SCROLL_RIGHT>":   {"type": "scroll", "dir": "right"},
    "TEXT_INPUT_SEQUENCE": {"type": "text_seq"},
    "<END_ACTION>":     {"type": "end"},
}

# ================= LOGIC WITH REASONING =================

def generate_action_sample(token_key, meta):
    """Generates (Instruction, Reasoning, ActionToken) triplet."""
    t_type = meta["type"]
    
    if t_type == "move":
        verb = random.choice(VOCAB["move_verb"])
        direction = random.choice(VOCAB[meta["dir"]])
        distance_desc = random.choice(VOCAB[f"{meta['dist']}_desc"])
        instr = f"{verb} {direction} {distance_desc}"
        
        # Semantic Reasoning
        reasoning = f"Reasoning: The user wants to {verb.lower()} the cursor {direction}. The distance described as '{distance_desc}' implies a {meta['dist']} range movement. The correct action is {token_key}."
        return instr, reasoning, token_key

    elif t_type == "click":
        target = random.choice(VOCAB["ui_element"])
        verb = random.choice(VOCAB["click_verb"])
        instr = f"{verb} the {target}"
        mode = "short click" if meta["mode"] == "short" else "long press"
        reasoning = f"Reasoning: The instruction is to {verb.lower()} on a UI element. This requires a {mode} interaction at the current position."
        return instr, reasoning, token_key

    elif t_type == "scroll":
        verb = "Scroll"
        direction = meta["dir"]
        instr = f"{verb} {direction}"
        reasoning = f"Reasoning: The user wants to navigate the view by scrolling {direction}. The corresponding action token is {token_key}."
        return instr, reasoning, token_key
    
    elif t_type == "text_seq":
        content = random.choice(VOCAB["text_content"])
        instr = f"Type '{content}'"
        response = f"<TEXT_START> {content} <TEXT_END>"
        reasoning = f"Reasoning: The user wants to input the text '{content}'. I need to wrap this content with text start and end tokens."
        return instr, reasoning, response

    elif t_type == "end":
        instr = random.choice(VOCAB["end_action"])
        reasoning = "Reasoning: The task is described as complete. I should terminate the session."
        return instr, reasoning, token_key

    return "Perform action", "Reasoning: Default action.", token_key

def generate_explanation_sample(token):
    """Generates definition Q&A."""
    question = f"[Define] What does {token} mean?"
    # Simple definition logic
    if "MOVE" in token:
        parts = token.split("_")
        direction = parts[1].lower()
        dist = parts[2].replace(">", "").lower()
        definition = f"It moves the cursor {direction} by a {dist} distance."
    elif "CLICK" in token:
        definition = random.choice(("It performs a click interaction.", "It means to click the current position."))
    elif "SCROLL" in token:
        if "UP" in token: definition = random.choice(("It is a control token. It means to scroll up.", "It means scrolling up at the current position."))
        elif "DOWN" in token: definition = random.choice(("It is a control token. It means to scroll down.", "It means scrolling down at the current position."))
        elif "LEFT" in token:
            definition = random.choice(("It is a control token. It means to scroll left.", "It means scrolling left at the current position."))
        elif "RIGHT" in token:
            definition = random.choice(("It is a control token. It means to scroll right.", "It means scrolling right at the current position."))
    elif "GO_BACK" in token:
        definition = random.choice(("It navigates back to the previous view.", "It means getting back to the previous page."))
    elif "GO_HOME" in token:
        definition = random.choice(("It navigates to the app's home screen.", "It means using the home button."))
    elif "TEXT"  in token:
        definition = "It indicates a sequence of user text input, typically wrapped with start/end text tokens."
    elif "END" in token:
        definition = random.choice(("It marks the end of an action sequence or session.", "It means stopping the action here."))
    else:
        definition = "It is a control token corresponding to a UI operation."
    
    reasoning = f"Reasoning: The user is asking for the definition of the token {token}. I will provide its functional description."
    return question, reasoning, definition

# ================= MAIN =================

def main():
    print(f"Generating Data with Reasoning...")
    all_records = []
    
    # 1. Action Execution (Semantic)
    for token_key, meta in TOKEN_META.items():
        for _ in range(SAMPLES_PER_ACTION):
            instr, reasoning, resp = generate_action_sample(token_key, meta)
            
            # Format: Reasoning + \n + Action: Token
            full_response = f"{reasoning}\nAction: {resp}"
            
            record = {
                "data_type": "action", # Needs Image (Noise)
                "messages": [
                    {"role": "user", "content": f"[Action] {instr}"},
                    {"role": "assistant", "content": full_response}
                ]
            }
            all_records.append(record)
            
    # 2. Explanation (Text Only)
    for token in TOKEN_META.keys():
        for _ in range(SAMPLES_PER_EXPLANATION):
            q, reasoning, a = generate_explanation_sample(token)
            full_response = f"{reasoning}\nAction: {a}" # Using Action format for consistency, though it's text
            
            record = {
                "data_type": "explanation", # Black Image
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": full_response}
                ]
            }
            all_records.append(record)
    
    random.shuffle(all_records)
    print(f"Writing {len(all_records)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()