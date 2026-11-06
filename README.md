# DrugDiscoveryIntelligence

Streamlit app that builds a disease-focused drug discovery report from Open Targets molecular targets and PubMed literature, with optional OpenAI summaries.

## Features

- Resolve a disease name to an Open Targets (EFO) identifier via GraphQL
- Pull up to 50 associated molecular targets with association scores
- Retrieve up to 20 PubMed articles (Medline) for “{disease} mechanisms drug discovery”
- Summarize each abstract with ChatOpenAI (when an API key is configured)
- Produce an aggregated meta-summary across those article summaries
- Render a markdown report with target list, PubMed links, and literature overview
- Sidebar shows Python version and whether an OpenAI API key is configured

## Requirements

See `requirements.txt`:

- streamlit
- requests
- openai
- langchain
- langchain_openai
- biopython
- pandas

## Setup

1. Clone the repo and create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Provide an OpenAI API key via environment variable `OPENAI_API_KEY` or Streamlit secrets key `OPENAI_API_KEY`.
4. Optionally set `ENTREZ_EMAIL` for NCBI Entrez (defaults to `researcher@example.com` if unset).

## Usage

```bash
streamlit run DiseaseSpecificDrugDiscoveryIntelligence.py
```

Enter a disease name (for example, Alzheimer’s disease or Breast Cancer) and click **Generate Comprehensive Report**.

## Notes

- Open Targets GraphQL endpoint used: `https://api.platform.opentargets.org/api/v4/graphql`
- Without an LLM key, abstract summaries fall back to a placeholder / original abstract text and meta-summary notes that the LLM is not configured.
