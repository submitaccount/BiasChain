from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, List

# Ideas:
# 1. Writing style and content bias
# 2. Topic and methodology bias
# 3. Confirmation bias
# 4. Bias from prior knowledge
# 5. Section & aspect bias


# Ideas revised:
# 1. Novelty Overemphasis Bias
# ----> Overvaluing the novelty of an approach at the expense of its practical or theoretical contribution.
# Example - A common tendency to prioritize novel techniques or findings even when incremental work may have higher long-term value.

# 2. Methodology Preference bias
# ----> Bias toward particular methodologies or topics that are currently dominant or trendy.
# Example - Reviewers may unconsciously favor papers that align with their own research interests or trending methods.

# 3. Confirmation bias
# ----> The tendency of reviewers to favor papers that align with their pre-existing beliefs, hypotheses, or prior work.
# Example - Reviewers may unintentionally favor research that confirms their existing views or approaches, even if other methodologies or perspectives are equally valid.

# 4. Publication Bias (Positive Results Bias)
# ----> A preference for papers with positive, significant, or state-of-the-art results.
# Example - Reviewers may prioritize works that report positive outcomes or breakthrough results, disregarding null or negative results.

# 5. Linguistic Proficiency Bias	
# ----> Penalization of authors for linguistic or writing quality deficiencies, particularly for non-native speakers.
# Example - Subtle biases arise when reviewers equate the fluency of writing with the quality of scientific content, especially for authors from non-English-speaking regions.



class ParentJsonSchema(BaseModel):
    bias_detected: Literal['True', 'False'] = Field(description="Is the bias detected?")
    bias_type: List[str] = Field(description="Types of biases detected from the provided tools strictly as returned by tools, if no bias from tools, write 'None'")
    confindence_score: float = Field(description="How sure are you about it overall?")
    evidence: List[str] = Field(description="In what statement did you find bias, if bias is detected else write 'None' if no bias found overall")
    suggestion_for_improvements: List[str] = Field(description="Give some improvement suggestions if bias is detected, else write 'None' if no bias found overall")
parent_parser = PydanticOutputParser(pydantic_object=ParentJsonSchema)

class ChildJsonSchema(BaseModel):
    bias_detected: Literal['True', 'False'] = Field("Is the bias detected?")
    bias_type: Literal['Novelty Bias', 'Confirmation Bias', 'Methodology Bias', 'Positive Results Bias', 'Linguistic Bias', 'None'] = Field(description="Name of the bias only if it is detected, else None")
    confindence_score: float = Field(ge=0, le=10, description="How sure are you about it?")
    evidence: str= Field(description="In what statement did you find bias, if bias is detected else write 'None'")
    suggestion_for_improvements: str= Field(description="Give some improvement suggestions if bias is detected, else write 'None'")
child_parser = PydanticOutputParser(pydantic_object=ChildJsonSchema)


parent_agent_prompt = PromptTemplate(
    template="""
You are an expert peer review bias detector.
But you do not think by yourself for the biases, you only rely on tools provided.
You have some tools provided to detect specific type of biases, which you have to use.

You are provided with:
Paper title, abstract (for context),
Original Review, Tone of the review, Justification behind classification of that tone,
Review consistency within itself, Explaination behind its classification,
Pairwise review comparison with other reviews of the same paper in which you have been provided:
    Is the review consistent with other reviews?
    Alignment Score with other reviews
    Contradictory points between the reviews if any
    Possible bias flags if any
    Summary of differences between the reviews if any

The input starts here:
**Paper Title**: {paper_title}
**Paper Abstract**: {paper_abstract}

**Original Review**: {original_review}
**Tone of the review**: {tone}
**Justification behind why this tone was classified**: {tone_reason}

**Review consistency within itself**: {consistency}
**Explaination behind why this consistency was classified**: {consistency_reason}

Pairwise Review comparison:
    **Is the review consistent with other reviews**: {is_consistent_with_others}
    **Alignment Score with other reviews**: {alignment_score}
    **Contradictory points between the reviews if any**: {contradictory_points}
    **Possible bias flags if any**: {possible_bias_flags}
    **Summary of differences between the reviews if any**: {summary_of_differences}


Your job is to study it thoroughly like a domain expert bias detector, use tools and classify it in the following schema:
{parent_json_schema}

Start!
""",
input_variables = ['paper_title', 'paper_abstract', 'original_review', 'tone', 'tone_reason',
                   'consistency', 'consistency_reason', 'is_consistent_with_others', 'alignment_score',
                   'contradictory_points', 'possible_bias_flags', 'summary_of_differences'],
partial_variables={'parent_json_schema': parent_parser.get_format_instructions()}
)


output_parser_prompt = PromptTemplate(
    template="""
You are an expert pydantic output parser that parses the output in the given schema.
Your task here is to parse the output into the schema given and remove unnecessary ```json and other things.
Just pure output schema, even if it is simple.
Do not add any extra information, keep yourself like RunnablePassthrough if needed but clean it.

The schema required:
{json_schema}


The input:
{input}
""",
    input_variables=['input'],
    partial_variables={'json_schema': parent_parser.get_format_instructions()}
)


novelty_bias_prompt = PromptTemplate(
    template="""
You are an expert in **novelty bias detection** in peer review texts.
Your task is to analyze the given peer review text and identify signs of bias related to the **overemphasis on novelty** over practical or theoretical contribution.
Your bias tool name is: Novelty Bias

Bias in this context may include:
- Excessive praise or criticism based solely on how "novel" the approach is
- Devaluation of incremental research or well-established methodologies
- Assumptions that novelty automatically equates to higher quality

You must return your findings in **strict adherence** to the following schema:
{child_json_schema}


The input starts here:
{input}
""",
    input_variables=["input"],
    partial_variables={"child_json_schema": child_parser.get_format_instructions()}
)

methodology_bias_prompt = PromptTemplate(
    template="""
You are an expert in **methodology preference bias detection** in peer review texts.
Your task is to evaluate whether the reviewer shows unjustified bias toward or against specific **methodologies or paradigms**.
Your bias tool name is: Methodology Bias

Bias in this context may include:
- Preference for specific frameworks, techniques, or tools regardless of their objective fit
- Dismissal of qualitative or alternative approaches in favor of dominant quantitative ones (or vice versa)
- Favoritism for trendy or mainstream methods without strong justification

You must return your findings in **strict adherence** to the following schema:
{child_json_schema}


The input starts here:
{input}
""",
    input_variables=["input"],
    partial_variables={"child_json_schema": child_parser.get_format_instructions()}
)

confirmation_bias_prompt = PromptTemplate(
    template="""
You are an expert in **confirmation bias detection** in peer review texts.
Your task is to identify whether the review shows **preference for findings or approaches that align with the reviewer's own beliefs, work, or assumptions**.
Your bias tool name is: Confirmation Bias

Bias in this context may include:
- Favorable evaluation of papers that support the reviewer’s previous work or perspectives
- Dismissal or skepticism toward alternative frameworks without objective critique
- Implicit reinforcement of the status quo or widely accepted theories without open-minded evaluation

You must return your findings in **strict adherence** to the following schema:
{child_json_schema}


The input starts here:
{input}
""",
    input_variables=["input"],
    partial_variables={"child_json_schema": child_parser.get_format_instructions()}
)

positive_results_bias_prompt = PromptTemplate(
    template="""
You are an expert in **publication bias detection**, particularly focused on the **favoring of positive or significant results** in peer review texts.
Your task is to identify whether the reviewer shows bias toward outcome-based evaluation, especially overvaluing positive, breakthrough, or state-of-the-art results.
Your bias tool name is: Positive Results Bias

Bias in this context may include:
- Disregard or undervaluation of null, negative, or replication studies
- Language implying that positive results are inherently more valuable or publishable
- Inflated praise for significant results without addressing methodological soundness

You must return your findings in **strict adherence** to the following schema:
{child_json_schema}


The input starts here:
{input}
""",
    input_variables=["input"],
    partial_variables={"child_json_schema": child_parser.get_format_instructions()}
)

linguistic_bias_prompt = PromptTemplate(
    template="""
You are an expert in detecting **linguistic proficiency bias** in peer review texts.
Your task is to assess whether the reviewer penalizes the author’s language use in ways that unfairly affect scientific evaluation.
Your bias tool name is: Linguistic Bias

Bias in this context may include:
- Overemphasis on grammar, fluency, or native-sounding English
- Equating writing quality with research quality, especially for non-native authors
- Non-constructive feedback focused on language rather than content clarity

You must return your findings in **strict adherence** to the following schema:
{child_json_schema}


The input starts here:
{input}
""",
    input_variables=["input"],
    partial_variables={"child_json_schema": child_parser.get_format_instructions()}
)




# # CONTENT
# content_bias_prompt = PromptTemplate(
#     template="""
# You are an expert in **writing style and content bias detection** in peer review texts.
# Your task is to examine the given peer review text and determine whether it contains writing-style-related or content-related bias.

# Bias in this context may include: 
# - Overly subjective language
# - Inconsistent or unprofessional tone
# - Dismissive or non-constructive feedback
# - Comments influenced more by writing style than technical merit

# You must return your findings in **strict adherence** to the following schema:
# {child_json_schema}


# The input starts here:
# {input}
# """,
# input_variables=['input'],
# partial_variables={'child_json_schema': child_parser.get_format_instructions()}
# )


# # TOPIC
# topic_bias_prompt = PromptTemplate(
#     template="""
# You are an expert in **topic and methodology bias detection** in peer review texts. 
# Your task is to analyze the given peer review text and determine if it contains bias based on:
# - The chosen topic of the research
# - The methodology or approach used (e.g., qualitative vs quantitative)
# - Reviewer preference for specific paradigms or frameworks over objective merit

# You must return your findings in **strict adherence** to the following schema:
# {child_json_schema}


# The input starts here:
# {input}
# """,
#     input_variables=["input"],
#     partial_variables={"child_json_schema": child_parser.get_format_instructions()}
# )


# # CONFIRMATION
# confirmation_bias_prompt = PromptTemplate(
#     template="""
# You are an expert in **confirmation bias detection** in peer review texts. 
# Analyze whether the reviewer is favoring information that confirms their pre-existing beliefs or prior expectations, while overlooking conflicting evidence or alternative interpretations.

# Common signs may include:
# - Selective praise or criticism
# - Dismissal of valid but contrary findings
# - Overemphasis on alignment with reviewer’s beliefs

# You must return your findings in **strict adherence** to the following schema:
# {child_json_schema}


# The input starts here:
# {input}
# """,
#     input_variables=["input"],
#     partial_variables={"child_json_schema": child_parser.get_format_instructions()}
# )


# # PRIOR KNOWLEDGE
# prior_knowledge_bias_prompt = PromptTemplate(
#     template="""
# You are an expert in detecting **bias from prior knowledge** in peer review texts. 
# Your task is to identify whether the reviewer is biased due to previous familiarity with the similar work.

# This may be reflected by:
# - Implicit references to previous works
# - Unjustified assumptions about the author's intent or capabilities
# - Overly confident judgment without adequate explanation

# Return your findings using the following schema:
# {child_json_schema}


# The input starts here:
# {input}
# """,
#     input_variables=["input"],
#     partial_variables={"child_json_schema": child_parser.get_format_instructions()}
# )


# output_parser_prompt = PromptTemplate(
#     template="""
# You are an expert pydantic output parser that parses the output in the given schema.
# Your task here is to parse the output into the schema given and remove unnecessary ```json and other things.
# Just pure output schema, even if it is simple.
# Do not add any extra information, keep yourself like RunnablePassthrough if needed but clean it.

# The schema required:
# {json_schema}


# The input:
# {input}
# """,
#     input_variables=['input'],
#     partial_variables={'json_schema': parent_parser.get_format_instructions()}
# )