# utils.py
"""
Module: utils.py
Description: function for evaluating stories and logging results.
"""
from agents import FactCheckerAgent

# A set of fantasy-related keywords to gauge fantasy element usage
FANTASY_KEYWORDS = ["dragon", "magic", "wizard", "kingdom", "sword", "castle", "prophecy", "elf", "ancient", "throne"]

def evaluate_story(story: str) -> dict:
    """
    Evaluate the given story text and return various metrics:
    - length: number of words in the story.
    - num_characters: number of distinct character names detected.
    - num_fantasy_elements: number of distinct fantasy keywords used.
    - consistency_issues: number of consistency issues detected in the story.
    - dialogue_lines: rough count of lines of dialogue (by detecting quotation marks).
    """
    metrics = {}
    # Word count as story length
    words = story.split()
    metrics['length'] = len(words)
    # Detect distinct character names (naive approach: capitalized words in text)
    import re
    name_candidates = re.findall(r'\b[A-Z][a-z]+\b', story)
    # Filter out any capitalized words that are at start of sentences or common words (simple heuristic)
    # For simplicity, assume all found capitalized words are character names or proper nouns
    characters = set(name_candidates)
    metrics['num_characters'] = len(characters)
    # Count distinct fantasy elements used
    used_fantasy = set()
    text_lower = story.lower()
    for term in FANTASY_KEYWORDS:
        if term in text_lower:
            used_fantasy.add(term)
    metrics['num_fantasy_elements'] = len(used_fantasy)
    # Use FactCheckerAgent to count consistency issues in the final story
    fact_checker = FactCheckerAgent()
    issues_count = 0
    sentences = re.split(r'[\.!?]+', story)
    story_so_far = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        story_so_far += sentence + ". "
        _, issues = fact_checker.check_consistency(sentence, story_so_far)
        issues_count += len(issues)
    metrics['consistency_issues'] = issues_count
    # Estimate number of dialogue lines by counting occurrences of quotation marks
    metrics['dialogue_lines'] = story.count('"') // 2 
    return metrics
