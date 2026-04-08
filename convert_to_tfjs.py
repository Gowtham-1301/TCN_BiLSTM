#!/usr/bin/env python3
"""
convert_to_tfjs.py — Export trained model to TensorFlow.js
===========================================================
Usage
-----
  python convert_to_tfjs.py
  python convert_to_tfjs.py --model ./models/ecg_beat_classifier.h5
  python convert_to_tfjs.py --model ./models/ecg_beat_classifier.h5 --output ./models/tfjs_model

Author : PULSE AI Team — KCG College of Technology
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.converter import TFJSConverter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

def main(args):
    conv = TFJSConverter(output_dir=args.output)
    ok   = conv.convert(args.model)
    if ok:
        conv.validate()
        conv.write_integration_snippet()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="./models/ecg_beat_classifier.h5")
    parser.add_argument("--output", default="./models/tfjs_model")
    main(parser.parse_args())
