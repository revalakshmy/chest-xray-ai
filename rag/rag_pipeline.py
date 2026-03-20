from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

#Loading documents
loader=PyPDFLoader("rag/knowledge_base/who_pneumonia.pdf")
documents=loader.load()
print(f"Loaded {len(documents)} pages")

#Splitting into chunks
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
# chunks=splitter.split_documents(documents)
# print(f"No of chunks: {len(chunks)}")

#Embeddings: chunks -> converted to vector so that it can get stored in the vector base

# Step 3: Create embeddings and vector store
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vector_store = FAISS.from_documents(chunks, embeddings)
# print("Vector store created!")

# vector_store.save_local("rag/vector_store")
# print("Vector Store saved!")

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#Scraping data
from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://www.nhs.uk/conditions/pneumonia/treatment/",
    "https://www.cdc.gov/pneumonia/about/index.html",
    "https://medlineplus.gov/pneumonia.html"
]

loader = WebBaseLoader(urls)
web_documents = loader.load()

all_documents=web_documents+documents

chunks=splitter.split_documents(all_documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store=FAISS.from_documents(chunks,embeddings)


if os.path.exists("rag/vector_store"):
    vector_store = FAISS.load_local("rag/vector_store", embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded!")
else:
    # build vector store from scratch
    vector_store.save_local("rag/vector_store")
    print("Vector Store built and saved from scratch!")



# Sample patient
patient_info = {
    "age": 75,
    "gender": "Male",
    "allergies": "Penicillin",
    "conditions": "None",
    "symptoms_duration": "3 days"
}

# function

def get_recommendation(diagnosis,confidence,patient_info):
    if(diagnosis=="Normal"):
        query = "pneumonia prevention respiratory health maintenance"
       
    else:
        query = f"Treatment guidelines for pneumonia patient with {confidence}% confidence diagnosis"
    
    relevant_chunks=vector_store.similarity_search(query,k=3)
    context="\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt=f"""You are a Medical Assistant AI for PneumoScan AI, a pneumonia screening tool.

    Do NOT recommend antibiotics for normal diagnosis.
    Focus on preventive care and reassurance only.

    Start with clear reassurance that no pneumonia was detected.

    Patient Information:
    - Age: {patient_info['age']}
    - Allergies: {patient_info['allergies']}
    - Existing conditions: {patient_info['conditions']}
    - Symptoms duration: {patient_info['symptoms_duration']}


    Patient Diagnosis: {diagnosis}
    Confidence Score: {confidence}%

    Medical Context from WHO Guidelines:
    {context}

    Based on the above WHO guidelines, provide:
    1. Immediate recommended actions
    2. Suggested medical tests
    3. Treatment approach
    4. Warning signs to watch for
    5. Follow-up timeline

    Be very specific about:
    1.Exact antibiotic names and dosages.
    2.Severity assessment based on confidence score.
    3.If confidence>80% consider it as a severe case.
    4.Specific actionable steps, not general advise.

    Important: Tailor recommendations based on the confidence score severity. 
    This is a screening tool - always recommend consulting a qualified doctor.


    WHO Guidelines context: {context}

    Provide personalized treatment recommendations considering patient's age, 
    allergies and existing conditions.
    """




    #Calling GROQ->LLM
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

# result = get_recommendation("Pneumonia", 85.74)
# print(result)

#Sample Test case
result = get_recommendation("Normal", 92, patient_info)
print(result)

