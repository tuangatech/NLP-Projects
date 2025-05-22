from langchain_core.prompts import PromptTemplate
from .config import prompt_template

# Create the prompt template using the string defined in config.py
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"]
)

# Format retrieved documents into a context string suitable for the LLM prompt
def format_context(docs):
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        section = doc.metadata.get("section_title", "Unknown Section")
        page = doc.metadata.get("end_page", "N/A")  # Using end_page instead of page_number
        content = doc.page_content.strip()
        block = f"[{i}] (Section: {section}, Page: {page})\n{content}"
        context_blocks.append(block)
    return "\n\n".join(context_blocks)

# Format chat history into a string suitable for the prompt
def format_chat_history(chat_history):
    if not chat_history:
        return ""
    lines = []
    for item in chat_history:
        lines.append(f"Q: {item['question']}")
        lines.append(f"A: {item['answer']}")
    return "\n".join(lines)

# Build the final prompt by filling the template with provided values
def build_prompt(context: str, chat_history: str, question: str) -> str:
    return prompt.format(
        chat_history=chat_history,
        context=context,
        question=question
    )