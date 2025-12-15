import re
from typing import Optional

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


ANSWER_MODE_SYSTEM_PROMPT = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

HINT_MODE_SYSTEM_PROMPT = """
You are a teaching assistant operating in HINT MODE. Never give the final answer.
Use only the provided class materials to:
- Point the student to where they should look (file from Source, slide/page number,
  or timestamp mentioned in the text).
- Briefly describe what to review there (1-2 sentences) and why it helps.
- Suggest 2-3 search keywords or prerequisite ideas to revisit.
If you lack an exact spot, propose the closest lecture/section from the retrieved sources. Keep the response concise.
"""

def get_quiz_question_prompt_template() -> PromptTemplate:
    """
    Prompt for generating exactly 3 numbered quiz questions without answers.
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a quiz writer. Using only the provided context, write EXACTLY 3 numbered quiz questions."
            "Output questions only, no answers. Keep each question concise.\n"
            "Context:\n{context}\n\nUser request:\n{question}\n\nQuestions:"
        ),
    )


def get_quiz_grading_prompt_template() -> PromptTemplate:
    """
    Prompt for grading a student's answer and providing verdict/correct answer/explanation.
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a teaching assistant checking a student's quiz answer using the provided context.\n"
            "Context:\n{context}\n\n{question}\n\n"
            "Provide:\n1) Verdict (Correct/Incorrect)\n2) Correct answer (1-3 sentences)\n"
            "Brief explanation: referencing the context(1-3 sentences).\n"
            "Keep the whole response very concise."
        ),
    )

def get_system_prompt(mode: str = "answer") -> str:
    """
    Return the system prompt based on mode. Defaults to the standard answer mode prompt.
    """
    if mode and mode.lower() == "hint":
        return HINT_MODE_SYSTEM_PROMPT
    return ANSWER_MODE_SYSTEM_PROMPT


def get_prompt_template(system_prompt: str = ANSWER_MODE_SYSTEM_PROMPT, promptTemplate_type=None, history: bool = False):
    if promptTemplate_type in ("qwen2.5", "qwen"):
        # Qwen2.5 ChatML-style headers
        B_SYS, E_SYS = "<|im_start|>system\n", "<|im_end|>\n"
        B_USER, E_USER = "<|im_start|>user\n", "<|im_end|>\n"
        B_ASSISTANT = "<|im_start|>assistant\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = (
                SYSTEM_PROMPT
                + B_USER
                + "Context: {history} \n {context}\nUser: {question}"
                + E_USER
                + B_ASSISTANT
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=instruction)
        else:
            instruction = (
                SYSTEM_PROMPT
                + B_USER
                + "Context: {context}\nUser: {question}"
                + E_USER
                + B_ASSISTANT
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=instruction)

    elif promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "llama3":

        B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
        B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> ", "<|eot_id|>"
        ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    print(f"Here is the prompt used: {prompt}")

    return (
        prompt,
        memory,
    )


_REWRITE_PROMPTS = {
    # retrieval-optimized rewrite for normal Q&A
    "answer": """You rewrite user queries for RETRIEVAL in a RAG system.
Goal: maximize retrieval recall/precision from lecture slides/transcripts/code/docs.
Rules:
- Do NOT answer the question.
- Preserve the user's intent and constraints.
- Expand acronyms and add key entities/keywords if implied.
- Remove filler words; keep it short.
- Output ONLY the rewritten query inside <rewritten>...</rewritten>.

User query:
{query}
""",
    # hint mode: retrieve "where to look", not solutions
    "hint": """You rewrite user queries for RETRIEVAL in a HINT-ONLY RAG system.
Goal: retrieve materials that help locate relevant sections (slides/pages/timestamps), not the final solution.
Rules:
- Do NOT solve the problem or provide the final answer.
- Preserve intent and constraints (unit/week/lecture if present).
- Add navigational retrieval keywords like: slide, transcript, timestamp, section, definition, example (only if helpful).
- Keep it short and search-friendly.
- Output ONLY the rewritten query inside <rewritten>...</rewritten>.

User query:
{query}
""",
    # quiz generation: retrieve key concepts for question creation
    "quiz_generation": """You rewrite user queries for RETRIEVAL to generate quiz questions from course materials.
Goal: retrieve core concepts, definitions, algorithms, common pitfalls, and examples for the requested topic/unit.
Rules:
- Do NOT generate quiz questions. Do NOT answer.
- Preserve unit/week/lecture constraints if present.
- Add course-typical keywords: definition, concept, algorithm, steps, example, pitfall, comparison (only if helpful).
- Output ONLY the rewritten query inside <rewritten>...</rewritten>.

User query:
{query}
""",
}


def get_rewrite_prompt(mode: str, submode: Optional[str] = None) -> str:
    """
    Fetch the rewrite prompt text for the given mode/submode, defaulting to the standard answer prompt.
    """
    key = submode or mode or "answer"
    return _REWRITE_PROMPTS.get(key, _REWRITE_PROMPTS["answer"])


def extract_rewritten(text: str) -> str:
    """
    Extract the rewritten query enclosed in <rewritten> tags; fall back to the first non-empty line.
    """
    m = re.search(r"<rewritten>\s*(.*?)\s*</rewritten>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else text.strip()
