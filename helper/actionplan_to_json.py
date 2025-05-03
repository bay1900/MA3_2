import re
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from helper import get_env, file_helper



def the_rag(): 

    
    KNOWLEDGE_BASE_PATH = get_env.retreive_value("KNOWLEDGE_BASE_PATH") # PRO 
    KNOWLEDGE_BASE_VECTOR_PATH = get_env.retreive_value("KNOWLEDGE_BASE_VECTOR_PATH")

    text_data = file_helper.read_txt(KNOWLEDGE_BASE_PATH)

    lines = text_data.split("\n")

    # Temporary storage for extracted sections before assigning genes
    metadata = []
    document_content = []
    current_topic = None
    current_gene = None
    min_age = None
    max_age = None
    
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("Topic:"):
            current_topic = line.split(":")[1].strip()
            i += 2
            continue

        # Identify and store gene (without assuming order)
        if line.startswith("Gene:"):
            current_gene = line.split(":")[1].strip()
            i += 2
            continue

        # Identify age range
        if line.startswith("Min Age:"):
            min_age = line.split(":")[1].strip()
            max_age = lines[i + 1].split(":")[1].strip()  # Next line contains Max Age
            # current_age_range = f"{min_age}-{max_age}"
            i += 3  # Skip max age line

        # Identify document sections (headings & content)
        if line.startswith("Heading:"):
            heading_1 = line.replace("Heading:", "").strip()
            heading_2 = ""
            text_lines = []

            if i + 1 < len(lines) and lines[i + 1].startswith("Heading:"):
                    # Consecutive headings
                    heading_2 = lines[i + 1].replace("Heading:", "").strip()
                    i += 2
            else:
                i += 1

            # Collect text until next blank line
            while i < len(lines) and lines[i].strip() != "":
                text_lines.append(lines[i])
                i += 1

            metadata.append({
                "Topic": current_topic,
                "Gene": current_gene,  # May need correction later
                "Min Age": min_age,
                "Max Age": max_age,
                "Heading 1": heading_1,
                "Heading 2": heading_2
            })

            document_content.append("\n".join(text_lines))

        i += 1
        
    print(metadata[5])
    print("\n")
    print(document_content[5])