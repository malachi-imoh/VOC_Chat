#!/usr/bin/env python3

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import logging
from tqdm import tqdm
import time
import os
from collections import defaultdict
import uuid
import sys
from datetime import datetime
from colorama import Fore, Style
import colorama
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI
import tiktoken


endpoint = "https://bhara-magvq2ls-eastus2.cognitiveservices.azure.com/"
deployment = "gpt-5-mini"   # your deployment name
api_version = "2024-12-01-preview"
subscription_key = os.getenv("AZURE_OPENAI_KEY")  # put your key in .env or Streamlit secrets

client = AzureOpenAI(
    api_key=subscription_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

colorama.init()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# This class is responsible for logging the output to a file
class Logger:
    def __init__(self, filename="gpt_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# This class is responsible for processing the Voice of Customer (VOC) data using a Map-Reduce pipeline
# We will sequentially process each question type, split the responses into batches, summarize each batch, and then create a meta-summary
import os
class VOCMapReduceProcessor:
    def __init__(
        self,
        persist_directory="chroma_database",
        batch_size: int = 60,
        openai_api_key=None
    ):
        # --- API Key handling: works with Streamlit Secrets or local .env ---
        if openai_api_key is None:
            if "OPENAI_API_KEY" in st.secrets:
                openai_api_key = st.secrets["OPENAI_API_KEY"]
            else:
                openai_api_key = os.getenv("OPENAI_API_KEY")

        



        """
        Initialize the Map-Reduce processor for VOC data.

        Args:
            persist_directory: Path to ChromaDB storage
            batch_size: Number of responses to process per batch
            anthropic_api_key: Anthropic API key for Claude
        """
        print(f"{Fore.GREEN}Initializing VOC Map-Reduce Processor{Style.RESET_ALL}")
        
        # This is the batch size (number of responses to process per batch)
        self.batch_size = batch_size

        # This is the order of which the questions will be processed
        self.question_order = [
            # Business Description Questions
            "desc_business_brief",
            "desc_primary_products",
            "desc_community_impact",
            "desc_equity_inclusion",
            
            # Business Obstacles and Goals
            "business_obstacles",
            "business_goals_1",
            "business_goals_2",
            
            # Financial Challenges Questions
            "financial_challenges_1",
            "financial_challenges_2",
            "financial_challenges_3",
            "financial_challenges_4",
            
            # Financial Tools and Needs
            "financial_tool_needs",
            "financial_advisor_questions",
            
            # Grant Related
            "grant_usage",
            "additional_context",

            # Newly added question types
            "reason_financial_assistance",
            "financial_planning_responsible"
        ]
        
        # These are the question types. These are the same as the keys in the question_types dictionary in the VOC_chroma_db_upload.py file
        self.question_types = {
            # Financial Challenges
            "financial_challenges_1": {
                "context": "What specific challenges do you face in managing and forecasting your cash flow?",
                "columns": ["What specific challenges do you face in managing and forecasting your cash flow?"]
            },
            "financial_challenges_2": {
                "context": "What specific financial tasks consume most of your time?",
                "columns": ["What specific financial tasks consume most of your time, and how do you feel these tasks impact your ability to focus on growing your business?"]
            },
            "financial_challenges_3": {
                "context": "Tell us about a hard instance managing finances or getting a loan",
                "columns": ["Please tell us about a recent instance where it was really hard for you to manage your finances, or to get financial help, such as a loan. What would have been the ideal solution?"]
            },
            "financial_challenges_4": {
                "context": "Challenges with applying for loans",
                "columns": ["What are the most significant challenges you face with applying for loans, and what do you wish you could improve?"]
            },

            # Business Description
            "desc_business_brief": {
                "context": "A brief description of the business",
                "columns": [
                    "Provide a brief description of your business",
                    "Provide a brief description of your business. Include a description of your products/services"
                ]
            },
            "desc_primary_products": {
                "context": "Primary products/services offered",
                "columns": ["Detail the primary products/services offered by your business"]
            },
            "desc_community_impact": {
                "context": "Impact on the community",
                "columns": ["Describe how your business positively impacts your community"]
            },
            "desc_equity_inclusion": {
                "context": "Efforts to promote equity and inclusion",
                "columns": ["Describe efforts made by your business to promote equity and inclusion in the workplace and community"]
            },

            # Business Goals and Growth
            "business_goals_1": {
                "context": "Achievements and business goals",
                "columns": [
                    "What significant achievements have you made in your business? What are your business goals for the coming year?",
                    "What significant achievements have you made in your business? What are your business goals for the next 12 months?"
                ]
            },
            "business_goals_2": {
                "context": "Daily tasks for a virtual CFO",
                "columns": ["If there were no constraints, what tasks would you want an advanced technology like a virtual Chief Financial Officer to handle for you daily?"]
            },

            # Financial Tools and Advisory
            "financial_tool_needs": {
                "context": "Required features for financial management tool",
                "columns": [
                    "What key features do you need in a tool to better manage your cash and build your business credit? What is (or would be) your budget for such a solution?",
                    "What key features do you need in a tool to better manage your cash and expenses? What is (or would be) your budget for such a solution?"
                ]
            },

            # Grant and Support
            "grant_usage": {
                "context": "How grant funds will be used",
                "columns": [
                    "Provide a brief statement detailing your financial need for this grant and how the funds will be used to enhance community impact",
                    "Provide a brief statement detailing how the funds will be used to enhance community impact"
                ]
            },

            # Business Challenges
            "business_obstacles": {
                "context": "Major business obstacles and solutions",
                "columns": ["Describe major obstacles your company encountered and how you resolved them"]
            },

            # Additional Context
            "additional_context": {
                "context": "Additional relevant information",
                "columns": ["Please include any relevant information or context that you believe would be helpful for the judges to consider when reviewing your application"]
            },

            # Financial Advisor Questions
            "financial_advisor_questions": {
                "context": "Questions for financial advisor",
                "columns": ["Please provide your top three (3) questions you would ask a financial advisor or business coach, about your business?"]
            },

            # Financial assistance rationale
            "reason_financial_assistance": {
                "context": "What is your main reason for seeking financial assistance for your business?",
                "columns": ["What is your main reason for seeking financial assistance for your business?"]
            },

            # Planning responsibility
            "financial_planning_responsible": {
                "context": "Who handles the financial planning and cash flow tracking at your business?",
                "columns": ["Who handles the financial planning and cash flow tracking at your business?"]
            }
        }
        
        # Here we are connecting to the ChromaDB database
        try:
            print(f"{Fore.YELLOW}Connecting to ChromaDB...{Style.RESET_ALL}")
            # Connection to the ChromaDB database
            self.client = chromadb.PersistentClient(path=persist_directory)
            # Using the 'voc_responses' collection with the SentenceTransformerEmbeddingFunction
            # Embedding function is used to convert text data into numerical vectors
            self.collection = self.client.get_collection(
                name="voc_responses",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name='all-MiniLM-L6-v2'
                )
            )
            print(f"{Fore.GREEN}Successfully connected to ChromaDB{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to connect to ChromaDB: {e}{Style.RESET_ALL}")
            raise

        # Initialize the Anthropic client for interacting with Claude
        self.openai_client = client
        self.deployment = deployment


    def get_question_types(self) -> List[str]:
        """
        Purpose: Get all unique question types from the ChromaDB collection.
        Input: None
        Output: List of unique question
        """
        # Get all documents and metadata from the collection
        results = self.collection.get()
        question_types = set()
        # For each metadata in the results, if the 'question_type' key exists, add it to the question_types set
        for metadata in results['metadatas']:
            if 'question_type' in metadata:
                # Only include question types that don't end with '_summary'
                if not metadata['question_type'].endswith('_summary'):
                    question_types.add(metadata['question_type'])
        return sorted(list(question_types))

    def get_responses_for_question(self, question_type: str) -> List[Dict]:
        """
        Purpose: Get all responses for a specific question type.
        Input: question_type (str)
        Output: List of dictionaries containing response text and metadata
        """
        # The output dictionary will contain the documents and metadata for the specified question type
        # We query the ChromaDB collection for documents where the 'question_type' key matches the input question_type
        results = self.collection.get(
            where={"question_type": question_type}
        )
        
        # For each document and metadata in the results, we create a dictionary with the 'text' and 'metadata' keys
        responses = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            responses.append({
                'text': doc,
                'metadata': meta
            })
        return responses

    def create_batch_summary_prompt(self, responses: List[Dict], question_type: str) -> str:
        """
        Purpose: Create a lightweight prompt for generation a batch summary. 
        Keeps it short so the model won't over generate and hit limit lengths. 
        Input: responses (List[Dict]), question_type (str)
        Output: Prompt for generating a batch summary as a string
        """
        context = self.question_types[question_type]['context']
        prompt = f"""You are analyzing Voice of Customer (VOC) responses for the question: {context}

        Summarrize the following{len(responses)} responses in 3 concise bullet points. 
        Keep the summary short, clear and focused on the most important themes.



        Responses to analyze:
        """
        # Enumerate gives us the index and the response in the responses list
        # We add the response text to the prompt
        ### hard cutoff for individual responses ###
        max_resp_tokens = 1000
        for i, resp in enumerate(responses, 1):
            text = resp['text']

            #count tokens for response
            tokens = self.count_tokens(text)
            if tokens > max_resp_tokens:
                print(f"{Fore.YELLOW}Truncating response {i} (was {tokens} tokens){Style.RESET_ALL}")

                #safe cutoff by words
                words = text.split()
                text = " ".join(words[:1000]) + "... [TRUNCATED]"

            prompt += f"\n{i}. {text}\n"
        
        print(f"{Fore.GREEN}Prompt for Batch Summary:\n{Style.RESET_ALL}{prompt}\n")
        return prompt

    def create_meta_summary_prompt(self, batch_summaries: List[str], question_type: str) -> str:
        """
        Purpose: Create a lightweight prompt for generating a meta-summary across multiple batch summaries. 
        Input: batch_summaries (List[str]), question_type (str)
        Output: Prompt for generating a meta-summary as a string
        """
        # Get the context for the question type
        context = self.question_types[question_type]['context']

        # Meta-summary prompt template
        prompt = f"""You are creating a comprehensive meta-analysis of all Voice of Customer (VOC) responses for the question: **{context}**

    Combine insights from {len(batch_summaries)} batch summaries into a single meta-summary.
    Present the key takeaways as 5 concise bullet points.
    - Focus only on the most important recurring themes and insights.
    - Avoid repeating details from each batch; just synthesize the overall picture.
    - Keep the response under 250 words.
    
    
        
        ---

        ### Batch Summaries to Synthesize:
        """
        for i, summary in enumerate(batch_summaries, 1):
            prompt += f"\nBatch {i}:\n{summary}\n"       

        print(f"{Fore.CYAN}Prompt for Meta-Summary:\n{Style.RESET_ALL}{prompt}\n")
        return prompt

    def get_llm_summary(self, prompt: str, prompt_tokens, force_max_tokens=None):
        """
        Purpose: Generate a summary using the OPENAI GPT model with infinite retries
        Input: prompt (str), initial_delay (int)
        Output: Summary generated by the model as a string
        """
    
        attempt = 1

        ##calculate prompt token length###
        prompt_tokens = self.count_tokens(prompt)
        max_context = 4096
        buffer = 100

        max_completion = max_context - prompt_tokens - buffer
        if force_max_tokens:
            max_completion = min(max_completion, force_max_tokens)
        if max_completion < 200:
            max_completion = 200
        elif max_completion > 1500:
            max_completion = 1500

        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment,
                max_completion_tokens=max_completion,
                messages=[{"role": "system", "content": "You are a concise summarizer. Keep your output under 700 tokens."},
                            {"role": "user", "content": prompt}
                    ]
                        
            )
        except Exception as e:
            print(f"{Fore.RED}[ERROR] OpenAI request failed: {e}{Style.RESET_ALL}")
            return None
                

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "length":
            print(f"{Fore.RED}[WARN] Truncated output (finish_reason=length). Retrying with smaller batch size or fewer tokens...{Style.RESET_ALL}")
            return self.get_llm_summary(prompt, prompt_tokens, force_max_tokens=800)
        
                    

                
        content = None
        try:
            content = choice.message.content
        except AttributeError:
            try:
                content = choice.messages[0].content
            except Exception:
                pass
                
        if not content:
            content = "[Empty] API returned no content."

        return content.strip()
            

    def store_summary(self, question_type: str, summary: str):
        """
        Purpose: Store the generated summary in the ChromaDB collection.
        Input: question_type (str), summary (str)
        Output: None
        """
        if not summary:
            summary = "[EMPTY] No final summary generated."
        try:
            # Create a unique document ID using the question type and a random UUID
            doc_id = f"{question_type}_summary_{uuid.uuid4()}"
            # Add the summary to the collection with metadata
            self.collection.add(
                documents=[summary],
                metadatas=[{
                    "question_type": f"{question_type}_summary",
                    "summary_type": "map_reduce",
                    "timestamp": time.time()
                }],
                ids=[doc_id]
            )
            print(f"{Fore.GREEN}Successfully stored summary for {question_type}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error storing summary: {e}{Style.RESET_ALL}")
            raise

    # def count_tokens(self, text: str, max_retries=3, delay=5) -> int:
    #     """
    #     Purpose: Helper function to count the number of tokens in a given text.
    #     Input: text (str), max_retries (int), delay (int)
    #     Output: Number of tokens in the text
    #     """
    #     for attempt in range(max_retries):
    #         try:
    #             response = self.openai_client.chat.completions.create(
    #                 model=self.deployment,
    #                 max_completion_tokens=1,
    #                 messages=[{"role": "user", "content": text}]
    #             )
    #             return response.usage.prompt_tokens
    #         except Exception as e:
    #             if attempt == max_retries - 1:
    #                 return 0
    #             print(f"{Fore.YELLOW}Server overloaded, retrying token count in {delay} seconds...{Style.RESET_ALL}")
    #             time.sleep(delay)
    #             delay *= 2
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens locally using tiktoken instead of calling the API.
        """
        try:
            enc = tiktoken.get_encoding("cl100k_base")  # works well for GPT models
            return len(enc.encode(text))
        except Exception:
            # fallback: rough estimate
            return len(text.split())






    def process_question_type(self, question_type: str):
        """
        Purpose: Process a specific question type using the Map-Reduce pipeline.
        Input: question_type (str)
        Output: Store the final summary in the ChromaDB collection
        """

        # === Step 1: Log which question type we are processing ===
        print(f"\n{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Processing question type: {question_type}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Context: {self.question_types[question_type]['context']}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
        
        # === Step 2: Retrieve responses for this question type ===
        responses = self.get_responses_for_question(question_type)
        if not responses:
            print(f"{Fore.YELLOW}No responses found for {question_type}. Skipping.{Style.RESET_ALL}")
            return

        # === Step 3: Split responses into batches ===
        batches = [responses[i:i + self.batch_size] for i in range(0, len(responses), self.batch_size)]
        print(f"{Fore.GREEN}Split {len(responses)} responses into {len(batches)} batches{Style.RESET_ALL}")

        # === Step 4: Process each batch (Map phase) ===
        batch_summaries = []
        for i, batch in enumerate(batches):
            print(f"\n{Fore.CYAN}Processing Batch {i+1}/{len(batches)}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}")
            
            # Show the raw responses included in this batch
            print(f"{Fore.YELLOW}Input Batch Contents:{Style.RESET_ALL}")
            for j, resp in enumerate(batch, 1):
                print(f"{Fore.YELLOW}{j}. {resp['text']}...{Style.RESET_ALL}")
            
            # Create the LLM prompt for summarizing this batch
            prompt = self.create_batch_summary_prompt(batch, question_type)

            token_count = self.count_tokens(prompt)
            print(f"{Fore.BLUE}Token count for prompt: {token_count}{Style.RESET_ALL}")

            ##if too large, split 
            max_prompt_tokens = 3000
            if token_count > max_prompt_tokens and len(batch) > 1:
                print(f"{Fore.YELLOW}Batch {i+1} too large ({token_count} tokens). Splitting recursively...{Style.RESET_ALL}")

                def split_and_process(sub_batch, depth=1):
                    sub_prompt = self.create_batch_summary_prompt(sub_batch, question_type)
                    sub_tokens = self.count_tokens(sub_prompt)

                    if sub_tokens > max_prompt_tokens and len(sub_batch) > 1:
                        mid = len(sub_batch) // 2
                        print(f"{Fore.YELLOW}Sub-batch at depth {depth} still too big ({sub_tokens}). Splitting again...{Style.RESET_ALL}")
                        split_and_process(sub_batch[:mid], depth + 1)
                        split_and_process(sub_batch[mid:], depth + 1)
                    else:
                        sub_summary = self.get_llm_summary(sub_prompt, self.count_tokens(sub_prompt))
                        if sub_summary:
                            batch_summaries.append(sub_summary)
                            self.store_summary(question_type, sub_summary)
                
                split_and_process(batch)
                continue


            # === Step 4a: Call the LLM with retry logic ===
            summary = None
            retry_delay = 15      # wait time between retries
            max_retries = 3       # maximum attempts
            attempt = 1

            while summary is None and attempt <= max_retries:
                try:
                    # Try to generate the batch summary
                    summary = self.get_llm_summary(prompt, self.count_tokens(prompt))
                except Exception as e:
                    print(f"{Fore.YELLOW}Batch {i+1} failed on attempt {attempt} with error: {e}{Style.RESET_ALL}")
                    if attempt < max_retries:
                        print(f"{Fore.YELLOW}Retrying Batch {i+1} in {retry_delay} seconds...{Style.RESET_ALL}")
                        time.sleep(retry_delay)
                        retry_delay *=2
                        attempt += 1
                        continue
                    else:
                        print(f"{Fore.RED}Skipping Batch {i+1} after {max_retries} failed attempts{Style.RESET_ALL}")
                        break

            # === Save batch summary (always attempt) ===
            if not summary:
                summary = "[EMPTY] No summary generated for this batch."

            print(f"\n{Fore.GREEN}Batch {i+1} Summary:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'-'*40}{Style.RESET_ALL}")
            print(summary)
            print(f"{Fore.GREEN}{'-'*40}{Style.RESET_ALL}")

            try:
                doc_id = f"{question_type}_batch_{i+1}_{uuid.uuid4()}"
                self.collection.add(
                    documents=[summary],
                    metadatas=[{
                        "question_type": f"{question_type}_batch_summary",
                        "batch_number": i+1,
                        "summary_type": "map",
                        "timestamp": time.time()
                    }],
                    ids=[doc_id]
                )
                print(f"{Fore.GREEN}STORING SUMMARY NOW: Batch {i+1} for {question_type}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error checkpointing batch {i+1}: {e}{Style.RESET_ALL}")

            batch_summaries.append(summary)


            print(f"{Fore.CYAN}Pausing 5s before next batch...{Style.RESET_ALL}")
            time.sleep(1)

        # === Step 5: Reduce Phase (meta-summary across all batch summaries) ===
        if batch_summaries:
            print(f"\n{Fore.BLUE}Creating meta-summary from {len(batch_summaries)} batch summaries{Style.RESET_ALL}")

            # Chunk batch summaries into groups
            chunk_size = 20
            chunks = [batch_summaries[i:i + chunk_size] for i in range(0, len(batch_summaries), chunk_size)]
            intermediate_summaries = []

            for k, chunk in enumerate(chunks, 1):
                print(f"\n{Fore.CYAN}Processing meta-chunk {k}/{len(chunks)} with {len(chunk)} batch summaries{Style.RESET_ALL}")

                meta_prompt = self.create_meta_summary_prompt(chunk, question_type)
                print(f"{Fore.CYAN}Prompt for Meta-Chunk {k}{Style.RESET_ALL}")

                meta_token_count = self.count_tokens(meta_prompt)
                print(f"{Fore.BLUE}Token count for meta-chunk prompt: {meta_token_count}{Style.RESET_ALL}")

                max_prompt_tokens = 3000
                if meta_token_count > max_prompt_tokens:
                    print(f"{Fore.YELLOW}Meta-chunk {k} too large ({meta_token_count} tokens). Splitting summaries...{Style.RESET_ALL}")

                    def split_meta_and_process(sub_summaries, depth=1):
                        sub_prompt = self.create_meta_summary_prompt(sub_summaries, question_type)
                        sub_tokens = self.count_tokens(sub_prompt)

                        if sub_tokens > max_prompt_tokens and len(sub_summaries) > 1:
                            mid = len(sub_summaries) // 2
                            print(f"{Fore.YELLOW}Sub-meta at depth {depth} too big ({sub_tokens}). Splitting again...{Style.RESET_ALL}")
                            split_meta_and_process(sub_summaries[:mid], depth + 1)
                            split_meta_and_process(sub_summaries[mid:], depth + 1)
                        else: 
                            sub_summary = self.get_llm_summary(sub_prompt, self.count_tokens(sub_prompt))
                            if sub_summary:
                                intermediate_summaries.append(sub_summary)
                                self.store_summary(question_type, sub_summary)
                        
                    split_meta_and_process(chunk)
                    continue


                # Pause before request
                print(f"{Fore.YELLOW}Pausing 30s before meta-chunk {k} request...{Style.RESET_ALL}")
                time.sleep(30)

                # Retry logic for meta-chunk
                summary = None
                retry_delay = 15
                max_retries = 3
                attempt = 1

                while summary is None and attempt <= max_retries:
                    try:
                        summary = self.get_llm_summary(meta_prompt, self.count_tokens(meta_prompt))
                    except Exception as e:
                        print(f"{Fore.YELLOW}Meta-Chunk {k} failed on attempt {attempt} with error: {e}{Style.RESET_ALL}")
                        if attempt < max_retries:
                            print(f"{Fore.YELLOW}Retrying Meta-Chunk {k} in {retry_delay} seconds...{Style.RESET_ALL}")
                            time.sleep(retry_delay)
                            attempt += 1
                            retry_delay *= 2 
                            continue
                        else:
                            print(f"{Fore.RED}Skipping Meta-Chunk {k} after {max_retries} failed attempts{Style.RESET_ALL}")
                            break

                if summary:
                    print(f"\n{Fore.GREEN}Meta-Chunk {k} Summary:{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}{'-'*40}{Style.RESET_ALL}")
                    print(summary)
                    print(f"{Fore.GREEN}{'-'*40}{Style.RESET_ALL}")
                    intermediate_summaries.append(summary)



            # Final reduce across intermediate summaries
            print(f"\n{Fore.BLUE}Creating FINAL meta-summary from {len(intermediate_summaries)} intermediate summaries{Style.RESET_ALL}")

            final_prompt = self.create_meta_summary_prompt(intermediate_summaries, question_type)
            final_token_count = self.count_tokens(final_prompt)
            print(f"{Fore.BLUE}Token count for final meta-summary prompt: {final_token_count}{Style.RESET_ALL}")

            max_prompt_tokens = 3000
            if final_token_count > max_prompt_tokens and len(intermediate_summaries) > 1:
                print(f"{Fore.YELLOW}Final meta-summary prompt too large ({final_token_count}). Splitting summaries...{Style.RESET_ALL}")

                def split_final_and_process(sub_summaries, depth=1):
                    sub_prompt = self.create_meta_summary_prompt(sub_summaries, question_type)
                    sub_tokens = self.count_tokens(sub_prompt)

                    if sub_tokens > max_prompt_tokens and len(sub_summaries) > 1:
                        mid = len(sub_summaries) // 2
                        print(f"{Fore.YELLOW}Sub-final at depth {depth} too big ({sub_tokens}). Splitting again...{Style.RESET_ALL}")
                        split_final_and_process(sub_summaries[:mid], depth + 1)
                        split_final_and_process(sub_summaries[mid:], depth + 1)
                    else:
                        sub_summary = self.get_llm_summary(sub_prompt, self.count_tokens(sub_prompt))
                        if sub_summary:
                            self.store_summary(question_type, sub_summary)

                # Run the recursive splitter instead of one giant request
                split_final_and_process(intermediate_summaries)
                return  # Skip the normal path since recursion handled it

            # Normal case: prompt is small enough
            final_summary = self.get_llm_summary(final_prompt, final_token_count)
            if not final_summary:
                final_summary = "[EMPTY] Failed to generate final meta-summary"

            print(f"\n{Fore.MAGENTA}FINAL META-SUMMARY FOR {question_type}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
            print(final_summary)
            print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")

            self.store_summary(question_type, final_summary)
######
            
####

    def process_all_questions(self):
        """Process only those question types which do not yet have a meta‐summary."""
        print(f"{Fore.GREEN}Determining which question types still need summaries{Style.RESET_ALL}")

        # 1) Find all existing meta‐summaries
        all_docs = self.collection.get()
        existing_summaries = {
            m["question_type"].rsplit("_summary", 1)[0]
            for m in all_docs["metadatas"]
            if (
                m.get("question_type", "").endswith("_summary")
                and m.get("summary_type") == 'map_reduce'
            )
        }

        # 2) Compute defined vs available vs to_do
        defined   = set(self.question_types.keys())
        available = set(self.get_question_types())
        to_do     = (defined & available) - existing_summaries

        print(f"{Fore.BLUE}Already have summaries for: {sorted(existing_summaries)}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Will generate summaries for: {sorted(to_do)}{Style.RESET_ALL}")

        # 3) Process in your defined order, only missing ones
        for q in self.question_order:
            if q in to_do:
                self.process_question_type(q)

        # 4) Any leftovers not in the explicit order
        leftovers = to_do - set(self.question_order)
        for q in sorted(leftovers):
            self.process_question_type(q)

    def cleanup_old_summaries(self):
        """
        Purpose: Remove all previous summary documents from the collection
        Input: None
        Output: None
        """
        try:
            print(f"{Fore.YELLOW}Cleaning up old summaries...{Style.RESET_ALL}")
            # Get all documents
            results = self.collection.get()
            
            # Find documents to delete (those with question_type ending in _summary)
            ids_to_delete = [
                doc_id for doc_id, metadata in zip(results['ids'], results['metadatas'])
                if metadata['question_type'].endswith('_summary')
            ]
            
            if ids_to_delete:
                print(f"{Fore.BLUE}Found {len(ids_to_delete)} summaries to delete{Style.RESET_ALL}")
                self.collection.delete(ids=ids_to_delete)
                print(f"{Fore.GREEN}Successfully deleted old summaries{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}No summaries found to delete{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error cleaning up summaries: {e}{Style.RESET_ALL}")
            raise

    def run_as_job(self):
        """
        Purpose: Run the processing as a continuous job that won't stop until complete
        Input: None
        Output: None
        """
        while True:
            try:
                print(f"\n{Fore.BLUE}Starting VOC Processing Job{Style.RESET_ALL}")
                self.process_all_questions()
                print(f"\n{Fore.GREEN}Job completed successfully!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error in job: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Retrying job in 60 seconds...{Style.RESET_ALL}")
                time.sleep(60)
                continue


def main():
    """Main function to run the map-reduce processing as a continuous job."""
    try:
        sys.stdout = Logger()
        
        print(f"\n{Fore.BLUE}Starting VOC Map-Reduce Processing Job{Style.RESET_ALL}\n")
        
        processor = VOCMapReduceProcessor(
            persist_directory="../chroma_db_q3_2025",
            batch_size=50,
        )


        processor.run_as_job()

        print(f"\n{Fore.GREEN}Job completed successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Fatal error in main: {e}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()
