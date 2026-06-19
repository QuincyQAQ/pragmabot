# Analysis Scripts

Standalone utilities for log inspection and paper figure generation. All scripts assume the top-level `requirements.txt` is installed and the `pragmabot` package is importable.

## Setup

```bash
# Activate the same virtual environment used for PragmaBot
source <catkin_workspace>/devel/setup.bash

# Only needed for evaluate_first_action.py
export OPENAI_API_KEY="your-api-key-here"
```

Run any script directly. Output goes to `scripts/output/`.

```bash
python3 <script_name>.py
```

Most scripts have a `FILENAME` or `INPUT_PATH` constant near the top — edit it to point at the file you want to process.

## Scripts

| Script | Description | Output |
| --- | --- | --- |
| `evaluate_first_action.py` | Run the full planning pipeline (scene description → LTM retrieval → action planning) on each image under `pragmabot/data/images/` and log results to Markdown. Requires VLM API credentials. | `output/<MODEL>_<TIMESTAMP>.md` |
| `convert_log_to_html.py` | Convert a Gradio JSON conversation log into a styled HTML page with compressed base64 images. | `output/<FILENAME>.html` |
| `extract_numbers_from_markdown.py` | Extract prompt token counts or VLM response times from a Markdown evaluation log. Toggle `count_token` to switch modes. | Console output |
| `plot_annotion_ablation.py` | Grouped bar charts comparing picking success rates and pushing distance errors with/without image annotation. | `output/annotation_ablation.pdf` |
| `plot_failure_flow.py` | Plotly Sankey diagram categorizing failures across 12 evaluation tasks. | `output/sankey_failure_analysis.html` |
| `plot_rag_ablation_spider_split.py` | Radar charts (GPT-4o vs. GPT-4o-mini) and bar charts (tokens, response time) for RAG retrieval ablation. | `output/rag_ablation_layout.pdf` |
