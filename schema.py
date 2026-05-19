from typing import TypedDict, List, Optional, Dict, Any
import os
from pydantic import BaseModel, Field


class State(TypedDict):
    contract_text: str                  # Raw input document
    extracted_clauses: List[Dict[str, Any]]  # Structured JSON clauses
    matched_policies: List[str]         # Compliance rules fetched from DB
    risk_analysis: List[Dict[str, Any]] # Auditor's findings
    loop_count: int                     # Infinite loop guardrail
    error_feedback: str                  # Message passed from Auditor back to Extractor for correction
    requires_human_review: bool         # Gateway flag
    is_audit_passed: bool              # Final audit result 


class ClauseExtraction(BaseModel):
    clause_title: str = Field(description="The name of the clause, e.g., Indemnification, Liability Limit")
    extracted_text: str = Field(description="The exact text snippet from the contract.")
    financial_value: float = Field(description="Any explicit monetary values found in this clause. Default to 0.0 if none.")

class ClauseExtractionSchema(BaseModel):
    clauses: List[ClauseExtraction]


# app = build_graph()

# result = app.invoke({
#     "contract_text": """
#     This agreement is between TechCorp and VendorX. 
#     Section 4: The total liability of VendorX under this agreement shall be capped at $1,000,000.
#     Section 9: This contract shall be governed by the laws of New York.
#     """
#     })

# print(result)