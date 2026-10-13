"""Cache the configured encoder before offline feature extraction."""
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from eot.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/eot_main.yaml')
args = parser.parse_args()
encoder = load_config(args.config).section('encoder')
options = {'revision': str(encoder['revision']), 'local_files_only': False}
AutoTokenizer.from_pretrained(str(encoder['model_name']), **options)
AutoModelForSequenceClassification.from_pretrained(str(encoder['model_name']), **options)
print('Configured tokenizer and encoder revision cached successfully.')
