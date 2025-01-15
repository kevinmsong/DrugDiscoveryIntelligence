import streamlit as st
import os
import socket
import requests
from openai import OpenAI
from langchain_openai import ChatOpenAI
from Bio import Entrez
import pandas as pd
import logging
import json
import io  # for parsing PubMed XML from memory

class APIConnectionLogger:
    def __init__(self):
        self.connection_logs = []

    def log_connection(self, api_name, status, details=None):
        log_entry = {
            'api': api_name,
            'status': status,
            'details': details or ''
        }
        self.connection_logs.append(log_entry)

    def display_logs(self):
        if not self.connection_logs:
            st.info("No API connection logs available.")
            return

        st.subheader("API Connection Logs")
        for log in self.connection_logs:
            status_color = "green" if log['status'] == "Success" else "red"
            st.markdown(f"**{log['api']}**: "
                        f"<span style='color:{status_color}'>{log['status']}</span>",
                        unsafe_allow_html=True)
            if log['details']:
                st.write(log['details'])


class DrugDiscoveryAgent:
    def __init__(self, api_logger):
        self.api_logger = api_logger
        
        # Attempt to retrieve your OpenAI API key
        try:
            self.openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets["OPENAI_API_KEY"]
        except Exception as e:
            api_logger.log_connection("OpenAI Key", "Failed", str(e))
            st.error("OpenAI API key not found. Please set in environment or Streamlit secrets.")
            self.openai_api_key = None

        # Initialize the LLM (if the key is available)
        if self.openai_api_key:
            try:
                self.llm = ChatOpenAI(
                    openai_api_key=self.openai_api_key,
                    temperature=0.1,
                    max_tokens=1024
                )
            except Exception as e:
                api_logger.log_connection("OpenAI", "Failed", str(e))
                st.error(f"Failed to initialize LLM: {e}")
                self.llm = None
        else:
            self.llm = None

        # Configure Entrez
        Entrez.email = "kmsong@uab.edu"

    def get_disease_gene_associations(self, disease_name):
        """
        Retrieve top 50 gene associations from OpenTargets,
        including a DNS resolution check for api.opentargets.org.
        """
        # 1) Attempt DNS resolution for api.opentargets.org
        try:
            socket.gethostbyname('api.opentargets.org')
        except Exception as dns_err:
            self.api_logger.log_connection(
                "OpenTargets",
                "DNS Resolution Failed",
                str(dns_err)
            )
            # Return empty list if DNS fails
            return []

        base_url = "https://api.opentargets.org/api/v4/graphql"
        try:
            # First, search for disease ID
            search_query = {
                "query": """
                query SearchDisease($diseaseName: String!) {
                    search(queryString: $diseaseName, entityNames: ["disease"], page: {index: 0, size: 1}) {
                        hits {
                            id
                            name
                        }
                    }
                }
                """,
                "variables": {"diseaseName": disease_name}
            }

            search_response = requests.post(base_url, json=search_query, timeout=10)
            search_data = search_response.json()

            hits = search_data.get('data', {}).get('search', {}).get('hits', [])
            if not hits:
                self.api_logger.log_connection(
                    "OpenTargets",
                    "No Disease Found",
                    f"No disease found for: {disease_name}"
                )
                return []

            disease_id = hits[0]['id']
            resolved_disease_name = hits[0]['name']

            # Query top 50 gene associations
            association_query = {
                "query": """
                query DiseaseGeneAssociations($diseaseId: String!) {
                    disease(efoId: $diseaseId) {
                        associatedTargets(page: {index: 0, size: 50}) {
                            rows {
                                target {
                                    id
                                    symbol
                                    name
                                    description
                                }
                                score
                            }
                        }
                    }
                }
                """,
                "variables": {"diseaseId": disease_id}
            }

            assoc_response = requests.post(base_url, json=association_query, timeout=10)
            assoc_data = assoc_response.json()

            associations = (
                assoc_data.get('data', {})
                          .get('disease', {})
                          .get('associatedTargets', {})
                          .get('rows', [])
            )

            if associations:
                self.api_logger.log_connection(
                    "OpenTargets",
                    "Success",
                    f"Retrieved {len(associations)} gene associations for {resolved_disease_name}"
                )
            else:
                self.api_logger.log_connection(
                    "OpenTargets",
                    "No Data",
                    f"No gene associations found for {resolved_disease_name}"
                )

            # Sort by descending score
            return sorted(associations, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            self.api_logger.log_connection("OpenTargets", "Failed", str(e))
            return []

    def retrieve_pubmed_literature_review(self, disease_name):
        """
        Retrieve top 20 PubMed articles with full details,
        including a short LLM-generated summary for each.
        """
        try:
            # Search PubMed using ESearch
            handle = Entrez.esearch(
                db="pubmed",
                term=f"{disease_name} mechanisms pathogenesis",
                sort="relevance",
                retmax=20
            )
            record = Entrez.read(handle)
            handle.close()

            # Log PubMed search results
            pmid_list = record.get('IdList', [])
            if pmid_list:
                self.api_logger.log_connection(
                    "PubMed",
                    "Success",
                    f"Retrieved {len(pmid_list)} articles for {disease_name}"
                )
            else:
                self.api_logger.log_connection(
                    "PubMed",
                    "No Data",
                    f"No articles found for {disease_name}"
                )
                return []

            # Fetch full article details directly
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid_list,
                rettype="xml",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            literature_review = []
            for record_item in records:
                if not isinstance(record_item, dict):
                    # Log and skip anything not in the expected dict format
                    self.api_logger.log_connection(
                        "PubMed",
                        "Article Processing Failed",
                        "Unexpected data type (not a dict)"
                    )
                    continue

                medline_citation = record_item.get('MedlineCitation', {})
                if not medline_citation:
                    continue

                article_details = medline_citation.get('Article', {})
                if not article_details:
                    continue

                title = article_details.get('ArticleTitle', 'No Title')
                pmid = medline_citation.get('PMID', 'Unknown PMID')

                abstract = 'No Abstract'
                abstract_data = article_details.get('Abstract')
                if abstract_data and 'AbstractText' in abstract_data:
                    # AbstractText can be list or string
                    atext = abstract_data['AbstractText']
                    if isinstance(atext, list) and atext:
                        abstract = atext[0]
                    elif isinstance(atext, str):
                        abstract = atext

                # Generate a short summary (if LLM is available)
                summary = "No LLM available for summary."
                if self.llm:
                    # Prompt for summarization
                    summary_prompt = f"""Please provide a concise scientific summary of this research article:
Title: {title}
Abstract: {abstract}

Focus on key findings, molecular mechanisms, and any novel insights related to {disease_name}.
"""
                    try:
                        summary_response = self.llm.invoke([
                            {"role": "system", "content": "You are a scientific researcher summarizing medical research articles."},
                            {"role": "user", "content": summary_prompt}
                        ])
                        summary = summary_response.content.strip()
                    except Exception as llm_err:
                        self.api_logger.log_connection("PubMed",
                                                       "LLM Summarization Failed",
                                                       str(llm_err))
                        summary = "Summary generation failed."

                literature_review.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'summary': summary
                })

            return literature_review

        except Exception as e:
            self.api_logger.log_connection("PubMed", "Failed", str(e))
            return []

    def generate_disease_report(self, disease_name):
        """
        Generate a comprehensive disease-specific report by integrating:
        - Gene associations from OpenTargets
        - Literature review from PubMed
        """
        try:
            # 1) Gene associations
            gene_associations = self.get_disease_gene_associations(disease_name)

            # 2) Literature review with LLM summaries
            literature_review = self.retrieve_pubmed_literature_review(disease_name)

            # Prepare text for final LLM prompt
            if gene_associations:
                gene_associations_text = (
                    "Top Gene Associations:\n" +
                    "\n".join([
                        f"- {assoc['target']['symbol']} (Score: {assoc['score']:.2f}): "
                        f"{assoc['target'].get('description', 'No description available')}"
                        for assoc in gene_associations[:10]  # top 10
                    ])
                )
            else:
                gene_associations_text = "No gene associations were retrieved."

            if literature_review:
                literature_insights = (
                    "Key Literature Insights:\n" +
                    "\n".join([
                        f"- PMID {article['pmid']}: {article['title']}\n  Summary: {article['summary']}"
                        for article in literature_review[:5]  # top 5
                    ])
                )
            else:
                literature_insights = "No recent literature data was retrieved."

            # If the LLM isn't available, just build a fallback text
            if not self.llm:
                fallback_report = (
                    f"Disease Name: {disease_name}\n\n"
                    f"{gene_associations_text}\n\n"
                    f"{literature_insights}\n\n"
                    "No LLM available to generate comprehensive report."
                )
                return {
                    'disease_name': disease_name,
                    'comprehensive_report': fallback_report,
                    'gene_associations': gene_associations,
                    'literature_review': literature_review
                }

            # Prompt for a final comprehensive report
            report_prompt = f"""Generate a comprehensive, evidence-based scientific report on {disease_name} that:
1. Integrates the following gene associations:
{gene_associations_text}

2. Incorporates insights from recent literature:
{literature_insights}

Report Requirements:
- Provide a detailed scientific overview
- Explicitly reference retrieved gene associations
- Incorporate findings from recent publications
- Discuss molecular mechanisms
- Explore potential therapeutic approaches
- Highlight current research landscape

Ensure the report is:
- Scientifically rigorous
- Up-to-date
- Comprehensive
- Directly citing retrieved genetic and literature data
"""

            try:
                comprehensive_report = self.llm.invoke([
                    {"role": "system", "content": "You are an expert medical researcher providing a detailed, data-driven disease report."},
                    {"role": "user", "content": report_prompt}
                ])
                final_report_text = comprehensive_report.content.strip()
            except Exception as e:
                self.api_logger.log_connection("Report Generation", "Failed", str(e))
                final_report_text = (
                    f"Could not generate final LLM report due to error: {str(e)}\n\n"
                    f"{gene_associations_text}\n\n{literature_insights}"
                )

            return {
                'disease_name': disease_name,
                'comprehensive_report': final_report_text,
                'gene_associations': gene_associations,
                'literature_review': literature_review
            }

        except Exception as e:
            self.api_logger.log_connection("Report Generation", "Failed", str(e))
            return {"error": str(e)}


def main():
    st.title("Disease-Specific Drug Discovery Intelligence")

    # Initialize API connection logger
    api_logger = APIConnectionLogger()

    # Instantiate the agent
    agent = DrugDiscoveryAgent(api_logger)

    disease_name = st.text_input("Enter a specific disease name:")

    if st.button("Generate Disease Report"):
        # Validate LLM initialization
        if not agent.llm:
            st.warning("LLM is unavailable. The final report will be partial.")
            # We can proceed, but we won't get a thorough summary.

        with st.spinner("Generating comprehensive disease report..."):
            results = agent.generate_disease_report(disease_name)

            # Display comprehensive report
            st.subheader(f"Comprehensive Report: {disease_name}")
            st.write(results.get('comprehensive_report', 'No report generated.'))

            # Display Gene Associations
            st.subheader("Key Gene Associations")
            gene_associations = results.get('gene_associations', [])
            if gene_associations:
                gene_df = pd.DataFrame([
                    {
                        'Gene Symbol': assoc['target']['symbol'],
                        'Gene Name': assoc['target']['name'],
                        'Association Score': assoc['score']
                    } 
                    for assoc in gene_associations
                ])
                st.dataframe(gene_df)
            else:
                st.write("No gene associations found.")

            # Display Literature Review
            st.subheader("Literature Review")
            literature_review = results.get('literature_review', [])
            for article in literature_review:
                st.markdown(f"**PMID {article['pmid']}: {article['title']}**")
                if 'summary' in article:
                    st.write(article['summary'])
                else:
                    st.write(article.get('abstract', "No abstract."))

            # Display API Connection Logs
            api_logger.display_logs()


if __name__ == "__main__":
    main()
