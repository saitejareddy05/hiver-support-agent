from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import SupportAgent, load_config
frame = pd.read_csv(Path(__file__).parents[1] / "data/processed/sample.csv")
agent = SupportAgent(frame, load_config(str(Path(__file__).parents[1] / "config.yaml")))
for text in ["Where is my package?", "I was charged twice and need a refund", "My account is hacked and this is urgent"]:
    result = agent.run(text)
    print("\nMESSAGE:", text, "\nINTENT:", result["intent"], f"({result['confidence']:.2f})", "\nREPLY:", result["reply"], "\nDECISION:", result["should_escalate"], result["reason"])
