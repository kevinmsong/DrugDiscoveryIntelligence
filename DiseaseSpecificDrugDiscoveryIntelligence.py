import streamlit as st
import os
import sys
import requests
from openai import OpenAI
from langchain_openai import ChatOpenAI
from Bio import Entrez, Medline
import logging
import time
import json

class DrugDiscoveryAgent:
    def __init__(self):
        # API Key and LLM initialization
        self.openai_api_key = os.getenv('OPENAI_API_KEY') or (
            st.secrets["OPENAI_API_KEY"] 
            if "OPENAI_API_KEY" in st.secrets 
            else None
        )
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            temperature=0.1,
            max_tokens=2048  # Adjust as needed for your summarization
        ) if self.openai_api_key else None

        # API configurations
        self.graphql_url = "https://api.platform.opentargets.org/api/v4/graphql"
        Entrez.email = os.getenv('ENTREZ_EMAIL', 'researcher@example.com')

    def _execute_graphql_query(self, query, variables=None):
        """Execute GraphQL query with error handling."""
        try:
            response = requests.post(
                self.graphql_url,
                json={"query": query, "variables": variables},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"GraphQL Query Error: {e}")
            return None

    def _get_disease_identifier(self, disease_name):
        """Resolve disease name to OpenTargets identifier."""
        search_query = """
        query($diseaseText: String!) {
          search(queryString: $diseaseText, entityNames: ["disease"], page: {index: 0, size: 1}) {
            hits { id name }
          }
        }
        """
        data = self._execute_graphql_query(search_query, {"diseaseText": disease_name})
        return data['data']['search']['hits'][0] if data and data['data']['search']['hits'] else None

    def _retrieve_associated_targets(self, disease_id):
        """Retrieve up to 50 associated molecular targets from OpenTargets."""
        disease_query = """
        query($diseaseId: String!) {
          disease(efoId: $diseaseId) {
            associatedTargets(page: {index: 0, size: 50}) {
              rows {
                target {
                  approvedSymbol
                  approvedName
                }
                score
              }
            }
          }
        }
        """
        data = self._execute_graphql_query(disease_query, {"diseaseId": disease_id})
        return data['data']['disease']['associatedTargets']['rows'] if data else []

    def _retrieve_pubmed_literature(self, disease_name):
        """
        Retrieve top 20 PubMed articles relevant to disease mechanism & drug discovery.
        Return a list of dicts with pmid, title, abstract, publication, and year.
        """
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=f"{disease_name} mechanisms drug discovery",
                sort="relevance",
                retmax=20
            )
            record = Entrez.read(handle)
            handle.close()
            pmid_list = record.get('IdList', [])

            literature_review = []
            for pmid in pmid_list:
                full_text_handle = Entrez.efetch(
                    db="pubmed",
                    id=pmid,
                    rettype="medline",
                    retmode="text"
                )
                record = Medline.read(full_text_handle)
                full_text_handle.close()

                literature_review.append({
                    'pmid': pmid,
                    'title': record.get('TI', 'No Title'),
                    'abstract': record.get('AB', 'No Abstract'),
                    'publication': record.get('JT', 'N/A'),
                    'year': record.get('DP', 'N/A')[:4]
                })

            return literature_review
        except Exception as e:
            st.error(f"PubMed retrieval error: {e}")
            return []

    def _summarize_abstract(self, title, abstract):
        """
        Summarize a single abstract using the LLM.
        If no LLM is configured, return an informative placeholder or the raw abstract.
        """
        if not self.llm:
            return "LLM not configured. Original abstract:\n" + abstract

        prompt_template = (
            "Please provide a concise scientific summary (2-3 sentences) of the following abstract. "
            "Emphasize methods or findings relevant to disease mechanisms and potential drug discovery.\n\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
        )
        try:
            response = self.llm(prompt_template)

            # --- Handle different possible return types ---
            if hasattr(response, "content"):
                # For ChatOpenAI or newer LangChain versions returning a ChatResult-like object
                return response.content.strip()
            elif isinstance(response, str):
                # Some versions or custom wrappers might return a plain string
                return response.strip()
            else:
                return "Summary unavailable: unexpected response format."
        except Exception as e:
            logging.error(f"Error summarizing abstract: {e}")
            return "Summary unavailable due to an LLM error."

    def _aggregate_summaries(self, summarized_literature):
        """
        Aggregate individual article summaries into a single combined text,
        then optionally create an LLM-based meta-summary of that combined text.
        """
        # Combine each short summary into a single text block
        combined_text = "\n\n".join(
            f"- Article {idx+1}: {item['abstract_summary']}"
            for idx, item in enumerate(summarized_literature)
        )

        if not self.llm:
            return "No LLM is configured. Unable to create a meta-summary."

        meta_prompt = (
            "Below are summaries of multiple articles related to disease mechanisms and drug discovery. "
            "Please provide a high-level (3-5 sentence) integrated overview mentioning any common themes, "
            "key methods, or notable outcomes.\n\n"
            f"{combined_text}\n\n"
            "Now, please provide your meta-summary:"
        )

        try:
            response = self.llm(meta_prompt)

            # --- Handle different possible return types ---
            if hasattr(response, "content"):
                # Typically a ChatResult object in newer LangChain versions
                return response.content.strip()
            elif isinstance(response, str):
                # If the LLM returns a plain string
                return response.strip()
            else:
                return "Meta-summary unavailable: unexpected response format."
        except Exception as e:
            logging.error(f"Error creating aggregated summary: {e}")
            return "Meta-summary unavailable due to an LLM error."

    def generate_disease_report(self, disease_name):
        """
        Generate a comprehensive drug discovery report that:
         - Identifies disease in OpenTargets.
         - Fetches top 50 targets.
         - Retrieves top 20 PubMed abstracts.
         - Summarizes each abstract individually.
         - Aggregates the summaries into a combined 'meta-summary'.
         - Incorporates the aggregated summary into the final report.
        """
        # Retrieve disease info
        disease_info = self._get_disease_identifier(disease_name)
        if not disease_info:
            return None

        # Retrieve data
        associated_targets = self._retrieve_associated_targets(disease_info['id'])
        literature_review = self._retrieve_pubmed_literature(disease_info['name'])

        # Summarize each article individually
        summarized_literature = []
        for article in literature_review:
            short_summary = self._summarize_abstract(article['title'], article['abstract'])
            summarized_literature.append({
                'pmid': article['pmid'],
                'title': article['title'],
                'year': article['year'],
                'publication': article['publication'],
                'abstract_summary': short_summary
            })

        # Generate an aggregated summary from all short summaries
        aggregated_summary = self._aggregate_summaries(summarized_literature)

        # Build the final report
        targets_section = self._format_targets(associated_targets)
        literature_section = self._format_summarized_literature(summarized_literature)

        report = f"""
# 🧬 Drug Discovery Intelligence: {disease_info['name']}

**Disease Name**: {disease_info['name']}

This report integrates molecular target data from OpenTargets and summarized findings 
from 20 relevant PubMed articles. Below, the top 50 targets are listed, followed by 
individual summaries of the key literature and an aggregated high-level overview.

## Top 50 Molecular Targets
{targets_section}

## Summarized PubMed Literature
{literature_section}

## Aggregated Overview of Literature Findings
{aggregated_summary}
"""
        return report

    def _format_targets(self, targets):
        """Format up to 50 molecular targets, sorted by descending score."""
        sorted_targets = sorted(targets, key=lambda x: x['score'], reverse=True)
        lines = []
        for i, t in enumerate(sorted_targets[:50], start=1):
            symbol = t['target']['approvedSymbol']
            name = t['target']['approvedName']
            score = t['score']
            lines.append(f"{i}. **{symbol}** (Score: {score:.2f}) — {name}")
        return "\n".join(lines)

    def _format_summarized_literature(self, summarized_literature):
        """
        Format the summarized literature, showing a hyperlink to PubMed, 
        plus the short LLM-based summary for each abstract.
        """
        items = []
        for i, article in enumerate(summarized_literature, start=1):
            title = article['title']
            pmid = article['pmid']
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            journal = article['publication']
            year = article['year']
            summary = article['abstract_summary']

            entry = (
                f"**{i}. [{title}]({link})**  \n"
                f"*Journal:* {journal} | *Year:* {year}  \n"
                f"**PMID:** {pmid}  \n"
                f"**Summary:** {summary}\n"
            )
            items.append(entry)
        return "\n".join(items)

def main():
    st.set_page_config(page_title="Drug Discovery Intelligence", page_icon="🧬")
    st.title("🔬 Drug Discovery Intelligence Report")

    # Initialize agent
    agent = DrugDiscoveryAgent()

    # Disease input
    disease_name = st.text_input(
        "Enter a disease name:", 
        help="E.g., Alzheimer's disease, Breast Cancer"
    )

    # Generate report button
    if st.button("Generate Comprehensive Report"):
        with st.spinner("Generating comprehensive report..."):
            report = agent.generate_disease_report(disease_name)
            if report:
                st.markdown(report)
            else:
                st.warning("Could not generate report.")

    # System debug info
    st.sidebar.title("🛠️ System Information")
    st.sidebar.write("Python Version:", sys.version)
    st.sidebar.write("OpenAI API Key:", 
                     "Configured" if agent.openai_api_key else "Not Configured")

if __name__ == "__main__":
    main()
