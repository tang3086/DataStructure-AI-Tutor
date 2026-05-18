# learning_log.py
import json
import os
from datetime import datetime
from pathlib import Path

class LearningLogger:
    def __init__(self, log_path: str = "learning_log.json"):
        self.log_path = Path(log_path)
        self.data = {
            "conversations": [],
            "exercises": [],
            "errors": [],
            "learning_profile": {
                "weak_topics": [],
                "strong_topics": []
            }
        }
        self._load()

    def _load(self):
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def _save(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def log_conversation(self, user_msg: str, assistant_msg: str, topics: list[str] = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_msg[:500],   # 截断过长的消息
            "assistant_response": assistant_msg[:500],
            "topics": topics or []
        }
        self.data["conversations"].append(entry)
        self._save()

    def log_exercise(self, question: str, user_answer: str, feedback: str, topic: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "question": question[:300],
            "user_answer": user_answer[:500],
            "feedback": feedback[:500]
        }
        self.data["exercises"].append(entry)
        self._save()

    def log_error(self, topic: str, description: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "description": description[:200]
        }
        self.data["errors"].append(entry)
        # 自动更新薄弱点
        if topic not in self.data["learning_profile"]["weak_topics"]:
            self.data["learning_profile"]["weak_topics"].append(topic)
        self._save()

    def update_profile(self, weak: list[str] = None, strong: list[str] = None):
        if weak:
            for w in weak:
                if w not in self.data["learning_profile"]["weak_topics"]:
                    self.data["learning_profile"]["weak_topics"].append(w)
        if strong:
            for s in strong:
                if s not in self.data["learning_profile"]["strong_topics"]:
                    self.data["learning_profile"]["strong_topics"].append(s)
        self._save()