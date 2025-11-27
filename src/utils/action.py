# src/utils/action.py

"""
GUI Agent Action Space Definition & Semantic Anchors.

This dictionary maps specific action tokens to a list of natural language words.
These words are used to initialize the embeddings of the action tokens, 
giving them a "Semantic Anchor" in the vector space.
"""

ACTION_BASE_EMBEDDING = {
    # ===========================
    # Movement (Cursor Control)
    # ===========================
    
    # Identifier for needing action currently.
    "<ACTION>": ["action", "do", "perform", "conduct", "act"],
    
    # Up
    "<MOVE_UP_FAR>":   ["move", "cursor", "up", "far", "high"],
    "<MOVE_UP_MID>":   ["move", "cursor", "up", "medium", "middle"],
    "<MOVE_UP_CLO>":   ["move", "cursor", "up", "close", "near", "tiny"],

    # Down
    "<MOVE_DOWN_FAR>": ["move", "cursor", "down", "far", "low"],
    "<MOVE_DOWN_MID>": ["move", "cursor", "down", "medium", "middle"],
    "<MOVE_DOWN_CLO>": ["move", "cursor", "down", "close", "near", "tiny"],

    # Left
    "<MOVE_LEFT_FAR>": ["move", "cursor", "left", "far", "west"],
    "<MOVE_LEFT_MID>": ["move", "cursor", "left", "medium", "middle"],
    "<MOVE_LEFT_CLO>": ["move", "cursor", "left", "close", "near", "tiny"],

    # Right
    "<MOVE_RIGHT_FAR>": ["move", "cursor", "right", "far", "east"],
    "<MOVE_RIGHT_MID>": ["move", "cursor", "right", "medium", "middle"],
    "<MOVE_RIGHT_CLO>": ["move", "cursor", "right", "close", "near", "tiny"],

    # ===========================
    # Interaction (Clicking)
    # ===========================
    "<CLICK_SHORT>": ["click", "tap", "select", "hit", "short"],
    "<CLICK_LONG>":  ["press", "hold", "long", "push", "down"],

    # ===========================
    # Navigation (System)
    # ===========================
    "<GO_BACK>": ["back", "return", "previous"],
    "<GO_HOME>": ["home", "main", "start", "desktop"],

    # ===========================
    # Scrolling
    # ===========================
    "<SCROLL_UP>":    ["scroll", "swipe", "up", "page"],
    "<SCROLL_DOWN>":  ["scroll", "swipe", "down", "page"],
    "<SCROLL_LEFT>":  ["scroll", "swipe", "left", "horizontal"],
    "<SCROLL_RIGHT>": ["scroll", "swipe", "right", "horizontal"],

    # ===========================
    # Text Input & Control
    # ===========================
    # Text tokens are often special; we anchor them to typing concepts
    "<TEXT_START>": ["start", "text", "type", "input", "keyboard"],
    "<TEXT_END>":   ["end", "text", "finish", "input", "enter"],
    
    # ===========================
    # Termination
    # ===========================
    "<END_ACTION>": ["end", "stop", "finish", "complete", "done", "terminate"],
}

# Helper list for easy access to keys
ACTION_TOKENS = list(ACTION_BASE_EMBEDDING.keys())