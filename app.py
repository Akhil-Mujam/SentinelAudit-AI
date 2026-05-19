from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from schema import State, ClauseExtractionSchema
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

# Vector DB & Free Embeddings
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()


# 2. DEFINE THE UNSTRUCTURED TEST CONTENT
# This replicates the content that would normally live inside your uploaded PDFs
COMPANY_POLICY_CONTENT = """
TechCorp Internal Compliance Guidelines v2
- Policy 1 (Financial Risk): The limitation of liability for any single standard software or service vendor agreement must never exceed $50,000. Any contract requesting higher liability limits must trigger a senior executive review.
- Policy 2 (Jurisdiction): All governing laws and legal dispute venues must belong strictly to the state of California or Delaware. No other state jurisdictions are permitted without explicit corporate counsel approval.
"""

TEST_VENDOR_CONTRACT = """
Standard Vendor Services Agreement
This agreement is entered into by TechCorp and CloudData Inc.
Section 4.1 (Liability Capping): In no event shall CloudData Inc.'s aggregate liability arising out of or related to this agreement exceed $1,000,000.
Section 9.7 (Governing Law): This agreement, and all claims or causes of action arising hereunder, shall be governed by and construed in accordance with the laws of the State of New York.
"""

# 3. SET UP FREE LOCAL VECTOR DATABASE (ChromaDB + Sentence Transformers)
# We use the free 'all-MiniLM-L6-v2' model from Sentence Transformers.
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="corporate_policies", embedding_function=local_ef)

# Seed our local vector database with the compliance policy data
collection.add(
    documents=[
        "Liability limit must never exceed $50,000 for standard vendors.",
        "All governing laws must belong strictly to the state of California or Delaware."
    ],
    ids=["policy_financial", "policy_jurisdiction"]
)

def get_llm():

    llm = ChatGroq(model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    api_key=os.getenv("GROQ_API_KEY"))

    return llm

def extract_clauses_node(state: State) -> State:
    """
    Extracts compliance-related legal clauses from unstructured contract text
    using Groq LLM with Pydantic-based structured output enforcement.

    This node identifies critical clauses such as liability, termination,
    confidentiality, payment terms, and governing law, then stores them
    as validated JSON objects in the shared graph state.

    Input:
        - contract_text

    Output:
        - extracted_clauses
    """

    print("--- NODE 1: EXTRACTING CLAUSES ---")

    llm = get_llm()

    prompt = PromptTemplate.from_template(
        """You are a legal expert specializing in contract analysis. 
        Your task is to extract key clauses from the provided contract text and structure them in a JSON format.   
        Contract Text:
        {contract_text} 
        Please identify and extract the following types of clauses:
        1. Indemnification: Clauses that specify the obligations of one party to compensate the other
        2. Liability Limit: Clauses that set a cap on the amount of damages one party can claim from the other
        3. Termination: Clauses that outline the conditions under which the contract can be terminated
        4. Confidentiality: Clauses that require parties to keep certain information confidential
        5. Payment Terms: Clauses that specify the payment schedule, amounts, and conditions
        6. Governing Law: Clauses that specify which jurisdiction's laws will govern the contract
        7. Dispute Resolution: Clauses that outline how disputes will be resolved, such as arbitration
        8. Force Majeure: Clauses that release parties from obligations in case of extraordinary events
        9. Intellectual Property: Clauses that address ownership and rights related to intellectual property
        10. Non-Compete: Clauses that restrict parties from engaging in competitive activities
        For each identified clause, provide the following information in a structured JSON format:
        - clause_title: The name of the clause (e.g., Indemnification, Liability Limit
        - extracted_text: The exact text snippet from the contract that corresponds to the clause
        - financial_value: Any explicit monetary values found in this clause. Default to 0.0
        Return the extracted clauses as a list of JSON objects under the key "extracted_clauses" in the state.

        {error_feedback}
        """
    )

    try:
        chain  = prompt | llm.with_structured_output(ClauseExtractionSchema) 

        response = chain.invoke({"contract_text": state['contract_text'], "error_feedback": state.get("error_feedback", "")})

        return {"extracted_clauses": [clause.model_dump() for clause in response.clauses], "error_feedback": "" }
    except Exception as e:
        print(f"Error during clause extraction: {e}")
        return {"extracted_clauses": [], "error_feedback": f"Extraction error: {str(e)}"}


def policy_router_node(state: State) :
    """
    Retrieves semantically relevant company compliance policies for each
    extracted contract clause using ChromaDB vector similarity search.

    This node converts extracted clauses into embedding search queries,
    matches them against internally indexed policy documents, and stores
    the most relevant compliance references in the shared graph state.

    Input:
        - extracted_clauses

    Output:
        - matched_policies
    """
    print("--- NODE 2: RETRIEVING COMPANY POLICIES ---")
    # In a full app, you would query ChromaDB here using state["extracted_clauses"]
    
    extracted_clauses = state["extracted_clauses"]
    matching_profiles = []

    for clause in extracted_clauses:
        query_text = f"{clause['clause_title']} {clause['extracted_text']}"
        results = collection.query(query_texts=[query_text], n_results=1)
        print(f"Queried ChromaDB with: {query_text}")
        print(f"ChromaDB returned: {results['documents']}")
        if results['documents'][0]:  # If we got a match
            matching_profiles.append(results['documents'][0][0])  # Add the top match


    return {"matched_policies": matching_profiles}

# NODE 3: The Auditor (Critique) Agent
def auditor_node(state: State):
    """
    Performs semantic compliance auditing by comparing extracted contract
    clauses against internally retrieved company policies using LLM-based analysis.

    This node evaluates potential policy violations, generates a structured
    risk summary, and determines whether the workflow requires human review
    escalation based on detected compliance conflicts.

    Input:
        - extracted_clauses
        - matched_policies
        - loop_count

    Output:
        - risk_analysis
        - requires_human_review
        - loop_count
    """
    print("--- NODE 3: AUDITING CLAUSES AGAINST POLICIES ---")

    clauses = state["extracted_clauses"]
    policies = state["matched_policies"]
    current_loops = state.get("loop_count", 0)
    llm = get_llm()
    
    prompt = f"""
    Compare these contract clauses to our company policies.
    Clauses: {clauses}
    Policies: {policies}
    
    Identify if any clause violates a policy. Return a summary.
    If a violation is found, state explicitly 'VIOLATION DETECTED'.
    """
    response = llm.invoke(prompt)
    analysis_text = str(response.content)
    
    # Check logic conditions
    requires_human = "VIOLATION DETECTED" in analysis_text
    
    return {
        "risk_analysis": [{"summary": analysis_text}],
        "requires_human_review": requires_human,
        "loop_count": current_loops + 1
    }
    

def routing_logic(state: State):

    if state["loop_count"] >= 3:
        print("\n!!! LOOP GUARDRAIL [ CIRCUIT BREAKER Condition] : Max loops reached. Escalating. !!!")
        return "human_gate"
    
    if state.get("error_feedback"):
        print("\n!!! ROUTING EDGE: Error feedback received. Routing to Extractor for correction. !!!")
        return "ExtractClauses"

    if state["requires_human_review"]:
        print("\n!!! ROUTING EDGE: Violation detected. Routing to Human Gate. !!!")
        return "human_gate"

    return "Approve"


def build_graph():
    graph = StateGraph(State)

    graph.add_node("ExtractClauses", extract_clauses_node)
    graph.add_node("PolicyRouter", policy_router_node)
    graph.add_node("Auditor", auditor_node)

    graph.add_edge(START, "ExtractClauses")
    graph.add_edge("ExtractClauses", "PolicyRouter")
    graph.add_edge("PolicyRouter", "Auditor")
    graph.add_conditional_edges("Auditor",
                    routing_logic,
                    {
                        "ExtractClauses" : "ExtractClauses",
                        "human_gate": END,
                        "Approve": END
                    })



    return graph.compile()



app = build_graph()

result = app.invoke({
    "contract_text": """
    This agreement is between TechCorp and VendorX. 
    Section 4: The total liability of VendorX under this agreement shall be capped at $1,000,000.
    Section 9: This contract shall be governed by the laws of New York.
    """
    })

print(f"Contract  Text : {result['contract_text']}")
print(f"Extracted Clauses : {result['extracted_clauses']}")
print(f"Matched Policies : {result['matched_policies']}")
print(f"Risk Analysis : {result['risk_analysis']}")
print(f"Requires Human Review? : {result['requires_human_review']}")
print(f"Loop Count : {result['loop_count']}")
print(f"Error Feedback : {result['error_feedback']}")
print(f"Final Audit Result (Pass/Fail): {'FAIL' if result['requires_human_review'] else 'PASS'}")