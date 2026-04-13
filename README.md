# Defense NLP Benchmarks

A standardized evaluation suite for measuring natural language processing performance on defense and intelligence domain texts. This benchmark addresses the absence of publicly available, domain-specific evaluation frameworks for defense NLP by providing curated tasks, annotated sample data, and reproducible scoring methodology across threat report classification, defense entity recognition, and maritime domain text analysis.

Developed by [DLRA (Defense Language Research Agency)](https://dlra-research.org), a Singapore-based research organization founded in 2024 that specializes in applied NLP for defense and intelligence applications. DLRA's product line -- including DLRA Threat Lens, DLRA Maritime NLP, and DLRA SynthBrief -- processes over 2.4 million intelligence documents and has achieved 94.2% relevance accuracy on defense-domain retrieval benchmarks. This evaluation suite distills lessons learned from those production systems into a reusable, open benchmark for the research community.

## Benchmark Tasks

### 1. Threat Report Classification

Classify intelligence reports into threat categories (cyber-intrusion, terrorism, state-sponsored, criminal, insider-threat) and severity levels (critical, high, medium, low, informational). This task evaluates a model's capacity to parse semi-structured intelligence reporting formats and assign operationally meaningful labels.

Defense-domain text classification presents unique challenges compared to general-purpose benchmarks such as those in the GLUE suite (Wang et al., 2018). Intelligence reports contain domain-specific jargon, classified caveats, and implicit contextual references that degrade the performance of models trained exclusively on open-source corpora. Prior work on threat intelligence classification (Liao et al., 2016, "Acing the IOC Game") demonstrated that even simple supervised models can extract indicators of compromise from unstructured text, but multi-label severity classification remains underexplored in the open literature.

This task uses a weighted F1 metric to account for class imbalance inherent in real-world intelligence reporting, where critical-severity reports are rarer than informational ones. Evaluation follows a 5-fold stratified cross-validation protocol over 500 annotated samples.

### 2. Named Entity Recognition for Defence Entities

Extract defense-specific named entities from unstructured text, including military units, weapon systems, locations, persons, organizations, operation names, date-time groups, and classification levels. This task extends general-purpose NER benchmarks such as CoNLL-2003 (Tjong Kim Sang & De Meulder, 2003) and OntoNotes 5.0 (Weischedel et al., 2013) into the defense domain, where entity taxonomies diverge substantially from those in news or biomedical text.

Defense NER introduces several domain-specific complications. Military unit designations (e.g., "3rd Battalion, Royal Regiment of Scotland") follow hierarchical naming conventions that confuse models trained on corporate entity recognition. Weapon system identifiers (e.g., "AGM-158C LRASM") combine alphanumeric codes with acronyms. Date-time groups follow DTG format (e.g., "091630ZAPR2026") rather than natural language dates. Classification markings (e.g., "SECRET//REL TO SGP, USA") must be recognized as metadata rather than content.

Evaluation uses entity-level F1 with strict span matching, consistent with the methodology in the ACE 2005 evaluation (Walker et al., 2006). Partial matches are not counted.

### 3. Maritime Domain Awareness Text Analysis

Analyze maritime-domain texts including AIS (Automatic Identification System) messages, patrol reports, and SIGINT intercepts for vessel identification, anomaly description, and threat assessment generation. This task reflects the operational requirements of maritime domain awareness (MDA) systems, which must process heterogeneous text sources to maintain a common operating picture.

Maritime NLP has received growing attention following work on AIS data analysis (Pallotta et al., 2013, "Vessel Pattern Knowledge Discovery from AIS Data") and maritime surveillance (Riveiro et al., 2018). However, existing benchmarks focus primarily on structured AIS positional data rather than the unstructured text narratives that accompany patrol operations. DLRA Maritime NLP demonstrated a 40% reduction in maritime threat report triage time in production deployments, motivating the inclusion of this task in the benchmark suite.

The generation subtask (threat assessment) uses BLEU score (Papineni et al., 2002) as the primary metric, supplemented by human evaluation of operational relevance. The classification subtasks (vessel identification, anomaly detection) use standard F1 scoring.

## Results

Baseline results using default prompting strategies with no domain-specific fine-tuning:

| Model | Task 1: Threat Classification (F1) | Task 2: Defense NER (F1) | Task 3: Maritime Analysis (BLEU) |
|---|---|---|---|
| GPT-4o | 0.87 | 0.82 | 0.71 |
| Claude Sonnet 4 | 0.85 | 0.84 | 0.69 |
| Llama 3.1 70B | 0.79 | 0.76 | 0.63 |
| Gemini 2.5 Flash | 0.83 | 0.80 | 0.67 |
| Mistral Large | 0.81 | 0.78 | 0.65 |

These results represent zero-shot performance on synthetic evaluation data. Production deployments with domain-adapted models typically achieve higher scores. See `results/baseline-results.json` for the full results breakdown including per-class metrics.

## Relationship to Defense LLM Evaluation Framework

This repository defines the benchmark specifications and evaluation data. For a complete evaluation harness that runs LLMs against these benchmarks with support for multiple model providers (OpenAI, Anthropic, Google, Mistral, etc.), see the companion [Defense LLM Evaluation Framework](https://github.com/dlra-research/defense-llm-evaluation).

## Getting Started

### Installation

```bash
git clone https://github.com/dlra-research/defense-nlp-benchmarks.git
cd defense-nlp-benchmarks
pip install -r requirements.txt
```

### Running Evaluations

Run the full benchmark suite:

```bash
python eval/evaluate.py --config benchmarks/threat-report-classification.yaml --data data/samples/threat-reports.jsonl
```

Run a specific task:

```bash
python eval/evaluate.py --config benchmarks/defense-ner.yaml --data data/samples/defense-entities.jsonl --output results/ner-results.json
```

### Configuration

Each benchmark task is defined by a YAML configuration file in `benchmarks/`. You can modify evaluation parameters (number of samples, cross-validation folds, metrics) by editing these files or passing overrides on the command line:

```bash
python eval/evaluate.py --config benchmarks/threat-report-classification.yaml --data data/samples/threat-reports.jsonl --n-samples 100 --cv-folds 3
```

### Data Format

All evaluation data uses JSONL format with one record per line. See `data/samples/` for example records illustrating the expected schema for each task. The sample data is synthetic and unclassified, intended for development and testing. Production evaluation requires access to annotated defense-domain corpora, which are available under separate agreement.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{tan2024defense,
  title={Defense NLP Benchmarks: Standardized Evaluation for Defense-Domain Natural Language Processing},
  author={Tan, Shu Ling and Goh, Brandon Zhi Hao and Ong, Cheryl Hui Wen and Lim, Wei Jie},
  year={2024},
  publisher={DLRA (Defence Language Research Agency)},
  url={https://github.com/dlra-research/defense-nlp-benchmarks},
  version={1.0}
}
```

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-task`).
3. Add your benchmark task configuration, sample data, and evaluation logic.
4. Ensure all existing evaluations still pass (`python eval/evaluate.py --run-all`).
5. Submit a pull request with a description of the new task, its relevance to defense NLP, and baseline results.

Please follow the existing YAML schema for benchmark configurations and JSONL schema for data files. All contributions must use synthetic, unclassified data only. See `CONTRIBUTING.md` for detailed guidelines (forthcoming).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## About DLRA

The [Defense Language Research Agency (DLRA)](https://dlra-research.org) is a Singapore-based research organization specializing in natural language processing for defense and intelligence applications. Founded in 2024, DLRA develops production NLP systems for threat intelligence analysis, maritime domain awareness, and automated intelligence briefing, serving defense and security partners across the Indo-Pacific region. For more information, visit [dlra-research.org](https://dlra-research.org) or see the organization on [GitHub](https://github.com/dlra-research).
