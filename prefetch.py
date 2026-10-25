"""Run at build time on Render so the model is downloaded once, not on first user request."""
from nlp import embed
print("Model ready:", embed(["warm up"]).shape)
