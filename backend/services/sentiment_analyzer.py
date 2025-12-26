# import os
# import json
# import asyncio
# from typing import List, Optional

# import httpx
# from transformers import pipeline
# from tenacity import retry, stop_after_attempt, wait_exponential


# class SentimentAnalyzer:
#     """
#     Unified interface for sentiment analysis using multiple model backends
#     """

#     def __init__(self, model_type: str = "local", model_name: str = None):
#         self.model_type = model_type

#         if model_type == "local":
#             self.sentiment_model_name = (
#                 model_name
#                 or os.getenv(
#                     "HUGGINGFACE_MODEL",
#                     "distilbert-base-uncased-finetuned-sst-2-english",
#                 )
#             )
#             self.emotion_model_name = os.getenv(
#                 "EMOTION_MODEL",
#                 "j-hartmann/emotion-english-distilroberta-base",
#             )

#             self.sentiment_pipeline = pipeline(
#                 "text-classification",
#                 model=self.sentiment_model_name,
#                 device=-1,  # CPU
#             )

#             self.emotion_pipeline = pipeline(
#                 "text-classification",
#                 model=self.emotion_model_name,
#                 device=-1,
#             )

#         elif model_type == "external":
#             self.api_key = os.getenv("EXTERNAL_LLM_API_KEY")
#             self.model_name = model_name or os.getenv("EXTERNAL_LLM_MODEL")
#             self.api_url = "https://api.groq.com/openai/v1/chat/completions"

#             if not self.api_key:
#                 raise ValueError("Missing EXTERNAL_LLM_API_KEY")

#             self.client = httpx.AsyncClient(
#                 headers={"Authorization": f"Bearer {self.api_key}"},
#                 timeout=30.0,
#             )
#         else:
#             raise ValueError("model_type must be 'local' or 'external'")

#     # ---------------- SENTIMENT ---------------- #

#     async def analyze_sentiment(self, text: str) -> dict:
#         if not text or not text.strip():
#             return {
#                 "sentiment_label": "neutral",
#                 "confidence_score": 0.0,
#                 "model_name": self.sentiment_model_name,
#             }

#         result = self.sentiment_pipeline(text[:512])[0]

#         label_map = {
#             "POSITIVE": "positive",
#             "NEGATIVE": "negative",
#         }

#         sentiment = label_map.get(result["label"], "neutral")

#         return {
#             "sentiment_label": sentiment,
#             "confidence_score": float(result["score"]),
#             "model_name": self.sentiment_model_name,
#         }

#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
#     async def _external_sentiment(self, text: str) -> dict:
#         prompt = f"""
# Return ONLY valid JSON.

# Analyze the sentiment of the text below.

# Text: "{text}"

# JSON format:
# {{
#   "sentiment_label": "positive|negative|neutral",
#   "confidence_score": 0.0-1.0
# }}
# """

#         response = await self.client.post(
#             self.api_url,
#             json={
#                 "model": self.model_name,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.2,
#             },
#         )

#         data = json.loads(response.json()["choices"][0]["message"]["content"])

#         return {
#             "sentiment_label": data["sentiment_label"],
#             "confidence_score": float(data["confidence_score"]),
#             "model_name": self.model_name,
#         }

#     # ---------------- EMOTION ---------------- #

#     async def analyze_emotion(self, text: str) -> dict:
#         if not text:
#             raise ValueError("Text cannot be empty")

#         if len(text.strip()) < 10:
#             return {
#                 "emotion": "neutral",
#                 "confidence_score": 0.0,
#                 "model_name": self.emotion_model_name,
#             }

#         result = self.emotion_pipeline(text)[0]

#         allowed = {"joy", "sadness", "anger", "fear", "surprise"}
#         emotion = result["label"].lower()

#         if emotion not in allowed:
#             emotion = "neutral"

#         return {
#             "emotion": emotion,
#             "confidence_score": float(result["score"]),
#             "model_name": self.emotion_model_name,
#         }

#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
#     async def _external_emotion(self, text: str) -> dict:
#         prompt = f"""
# Return ONLY valid JSON.

# Detect the strongest emotion in the text.

# Text: "{text}"

# Allowed emotions:
# joy, sadness, anger, fear, surprise, neutral

# JSON format:
# {{
#   "emotion": "...",
#   "confidence_score": 0.0-1.0
# }}
# """

#         response = await self.client.post(
#             self.api_url,
#             json={
#                 "model": self.model_name,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.2,
#             },
#         )

#         data = json.loads(response.json()["choices"][0]["message"]["content"])

#         return {
#             "emotion": data["emotion"],
#             "confidence_score": float(data["confidence_score"]),
#             "model_name": self.model_name,
#         }

#     # ---------------- BATCH ---------------- #

#     async def batch_analyze(self, texts: List[str]) -> List[dict]:
#         if not texts:
#             return []

#         results = []
#         for text in texts:
#             try:
#                 results.append(await self.analyze_sentiment(text))
#             except Exception as e:
#                 results.append({"error": str(e)})
#         return results

#         tasks = [self.analyze_sentiment(t) for t in texts]
#         return await asyncio.gather(*tasks, return_exceptions=True)









import os
import json
import asyncio
from typing import List

import httpx
from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_exponential


class SentimentAnalyzer:
    """
    Unified interface for sentiment analysis using multiple model backends
    """

    def __init__(self, model_type: str = "local", model_name: str = None):
        self.model_type = model_type

        # ---------------- LOCAL MODELS ---------------- #
        if model_type == "local":
            self.sentiment_model_name = (
                model_name
                or os.getenv(
                    "HUGGINGFACE_MODEL",
                    "distilbert-base-uncased-finetuned-sst-2-english",
                )
            )

            self.emotion_model_name = os.getenv(
                "EMOTION_MODEL",
                "j-hartmann/emotion-english-distilroberta-base",
            )

            self.sentiment_pipeline = pipeline(
                "text-classification",
                model=self.sentiment_model_name,
                device=-1,  # CPU
            )

            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.emotion_model_name,
                device=-1,
            )

        # ---------------- EXTERNAL (GROQ) ---------------- #
        elif model_type == "external":
            self.api_key = os.getenv("EXTERNAL_LLM_API_KEY")
            self.model_name = model_name or os.getenv("EXTERNAL_LLM_MODEL")
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"

            if not self.api_key:
                raise ValueError("Missing EXTERNAL_LLM_API_KEY")

            self.client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )

        else:
            raise ValueError("model_type must be 'local' or 'external'")

    # ================= SENTIMENT ================= #

    async def analyze_sentiment(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "sentiment_label": "neutral",
                "confidence_score": 0.0,
                "model_name": (
                    self.model_name if self.model_type == "external"
                    else self.sentiment_model_name
                ),
            }

        if self.model_type == "external":
            return await self._external_sentiment(text)

        result = self.sentiment_pipeline(text[:512])[0]

        label_map = {
            "POSITIVE": "positive",
            "NEGATIVE": "negative",
        }

        sentiment = label_map.get(result["label"], "neutral")

        return {
            "sentiment_label": sentiment,
            "confidence_score": float(result["score"]),
            "model_name": self.sentiment_model_name,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _external_sentiment(self, text: str) -> dict:
        prompt = f"""
Return ONLY valid JSON.

Analyze the sentiment of the text below.

Text: "{text}"

JSON format:
{{
  "sentiment_label": "positive|negative|neutral",
  "confidence_score": 0.0
}}
"""

        response = await self.client.post(
            self.api_url,
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
        )

        content = response.json()["choices"][0]["message"]["content"]
        content = content.strip().strip("```json").strip("```")
        data = json.loads(content)

        return {
            "sentiment_label": data["sentiment_label"],
            "confidence_score": float(data["confidence_score"]),
            "model_name": self.model_name,
        }

    # ================= EMOTION ================= #

    async def analyze_emotion(self, text: str) -> dict:
        if not text:
            raise ValueError("Text cannot be empty")

        if len(text.strip()) < 10:
            return {
                "emotion": "neutral",
                "confidence_score": 0.0,
                "model_name": (
                    self.model_name if self.model_type == "external"
                    else self.emotion_model_name
                ),
            }

        if self.model_type == "external":
            return await self._external_emotion(text)

        result = self.emotion_pipeline(text)[0]

        allowed = {"joy", "sadness", "anger", "fear", "surprise"}
        emotion = result["label"].lower()

        if emotion not in allowed:
            emotion = "neutral"

        return {
            "emotion": emotion,
            "confidence_score": float(result["score"]),
            "model_name": self.emotion_model_name,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _external_emotion(self, text: str) -> dict:
        prompt = f"""
Return ONLY valid JSON.

Detect the strongest emotion in the text.

Text: "{text}"

Allowed emotions:
joy, sadness, anger, fear, surprise, neutral

JSON format:
{{
  "emotion": "...",
  "confidence_score": 0.0
}}
"""

        response = await self.client.post(
            self.api_url,
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
        )

        content = response.json()["choices"][0]["message"]["content"]
        content = content.strip().strip("```json").strip("```")
        data = json.loads(content)

        return {
            "emotion": data["emotion"],
            "confidence_score": float(data["confidence_score"]),
            "model_name": self.model_name,
        }

    # ================= BATCH ================= #

    async def batch_analyze(self, texts: List[str]) -> List[dict]:
        if not texts:
            return []

        if self.model_type == "external":
            tasks = [self.analyze_sentiment(t) for t in texts]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for text in texts:
            try:
                results.append(await self.analyze_sentiment(text))
            except Exception as e:
                results.append({"error": str(e)})

        return results
