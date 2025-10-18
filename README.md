# [ACL 25] Value Portrait: Assessing Language Models’ Values through Psychometrically and Ecologically Valid Items


Repository for the following paper:

> **Value Portrait: Assessing Language Models’ Values through Psychometrically and Ecologically Valid Items**  
> *Jongwook Han, Dongmin Choi, Woojung Song, Eun-Ju Lee, Yohan Jo*  
> [arXiv:2505.01015](https://arxiv.org/pdf/2505.01015)

---

## 📦 Installation

```bash
# Clone this repository
git clone https://github.com/holi-lab/ValuePortrait.git
cd ValuePortrait

# (Optional) Create a new environment
conda create -n vp python=3.10
conda activate vp

# Install dependencies
pip install -r requirements.txt
```
---

## 🗂️ Data

The `data/` directory contains the ValuePortrait dataset. The dataset consists of query-response pairs with tagged with the 10 Schwartz values and the 5 personality traits from the Big Five Inventory (BFI-10).

### Tagged Data(`data/query-response-tagged/`)

| Dataset | Description | Format |
|---------|-------------|---------|
| `query-response-tagged.json` | Query-response pairs where each response is tagged with correlation scores to the 10 Schwartz values and 5 Big Five personality traits | JSON |

### Query-Response Data (`data/query-response/`)

| Dataset | Description | Format |
|---------|-------------|---------|
| `DearAbby.json` | Query-response pairs sourced from the Dear Abby advice column | JSON |
| `LMSYS.json` | Query-response pairs sourced from the LMSYS Chatbot Arena | JSON |
| `Reddit.json` | Query-response pairs sourced from the AITA subreddit | JSON |
| `ShareGPT.json` | Query-response pairs sourced from the ShareGPT dataset | JSON |

### Query Categories (`data/query/`)
| Dataset | Description | Format |
|---------|-------------|---------|
| `DearAbby.json` | Original queries sourced from the Dear Abby advice column | JSON |
| `LMSYS.json` | Original queries sourced from the LMSYS Chatbot Arena | JSON |
| `Reddit.json` | Original queries sourced from the AITA subreddit | JSON |
| `ShareGPT.json` | Original queries sourced from the ShareGPT dataset | JSON |
| `query_categorized.json` | Categorized queries by topic for systematic analysis | JSON |

### Correlation Analysis Results (`data/correlation_results/`)

| Dataset | Description | Format |
|---------|-------------|---------|
| `pvq_correlation_results.json` | Correlation analysis results on the 10 Schwartz values | JSON |
| `higher_pvq_correlation_results.json` | Higher-order Schwartz values correlation analysis results | JSON |
| `bfi_correlation_results.json` | Correlation analysis results on the 5 Big Five personality traits | JSON |

### Survey Data (`data/prolific/`)

| Dataset | Description | Format |
|---------|-------------|---------|
| `survey/main/survey.json` | Main survey responses from Prolific participants | JSON |
| `survey/pilot/` | Pilot survey data for preliminary analysis | JSON |
| `value/` | The value orientations and personality traits of the Prolific participants measured using the PVQ-21 and BFI-10 questionnaires | JSON |

---

## 🧾 Citation

If you use this work, please cite:

```bibtex
@article{han2025value,
  title={Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items},
  author={Han, Jongwook and Choi, Dongmin and Song, Woojung and Lee, Eun-Ju and Jo, Yohan},
  journal={arXiv preprint arXiv:2505.01015},
  year={2025}
}
```

## 🔑 License

This repository is released under the **MIT License**. See `LICENSE` for details.

---