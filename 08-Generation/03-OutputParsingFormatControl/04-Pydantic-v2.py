from pydantic import BaseModel, Field
from typing import List, Optional
from llama_index.program.openai import OpenAIPydanticProgram

# Define the code issue model
class CodeIssue(BaseModel):
    """An issue found in the code"""
    line_number: int = Field(..., description="Line number where the issue is located")
    issue_type: str = Field(..., description="Type of issue, e.g.: security vulnerability, performance issue, code style, etc.")
    description: str = Field(..., description="Detailed description of the issue")
    severity: str = Field(..., description="Issue severity: high/medium/low")

# Define the code analysis report model
class CodeAnalysis(BaseModel):
    """Code analysis report"""
    file_name: str = Field(..., description="Name of the file being analyzed")
    issues: List[CodeIssue] = Field(default_factory=list, description="List of discovered issues")
    overall_quality: str = Field(..., description="Overall code quality assessment: excellent/good/fair/poor")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")

# Create OpenAI Pydantic Program
program = OpenAIPydanticProgram.from_defaults(
    output_cls=CodeAnalysis,
    prompt_template_str="""
Please analyze the following code and generate a detailed analysis report:
{code}

Requirements:
1. Identify potential issues in the code
2. Evaluate code quality
3. Provide improvement recommendations
""",
    verbose=True
)

# Sample code
sample_code = """
def process_data(data):
    if data is None:
        return
    for item in data:
        if item > 100:
            print("Large value found")
        else:
            print("Small value")
"""

# Run analysis
try:
    analysis = program(code=sample_code)

    print(f"File analysis report: {analysis.file_name}")
    print(f"Overall quality: {analysis.overall_quality}")

    print("\nDiscovered issues:")
    for issue in analysis.issues:
        print(f"- Line {issue.line_number}: {issue.issue_type}")
        print(f"  Description: {issue.description}")
        print(f"  Severity: {issue.severity}")

    print("\nImprovement recommendations:")
    for rec in analysis.recommendations:
        print(f"- {rec}")

except Exception as e:
    print(f"Error during analysis: {e}")
