# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Step 1: Define the desired output schema using pydantic
class AnswerSchema(BaseModel):
    answer: str = Field(description="The corresponding record number surrounded by \"[]\" or \"[0]\" if there is none.")
    explanation: str = Field(description="A brief explanation of the answer.")

# Step 2: Create the output parser
parser = PydanticOutputParser(pydantic_object=AnswerSchema)

# Step 3: Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    # ("system", "You are a helpful assistant trained in Entity Matching."),
    ("human", "Select a record from the following candidates that refers to the same "\
              "real-world entity as the given record. \nGiven entity record:  "\
              "[COL] title [VAL] Benchmarking Spatial Join Operations with Spatial Output"\
              " [COL] authors [VAL] Erik G. Hoel, Hanan Samet [COL] venue [VAL] VLDB [COL] year "\
              "[VAL] 1995\n \nCandidate records:\n[1]  [COL] title [VAL] Benchmarking Spatial "\
              "Join Operations with Spatial Output [COL] authors [VAL] EG Hoel, H Samet [COL] venue "\
              "[VAL] PROCEEDINGS OF THE INTERNATIONAL CONFERENCE ON VERY LARGE  &hellip;, [COL] year"
              "[VAL] 1995.0\n[2]  [COL] title [VAL] Efficient Processing of Spatial Joins Using R-trees "\
              "[COL] authors [VAL] T Brinkhoff, HP Kriegel, B Seeger\n[3]  [COL] title [VAL] Approximations "\
              "for a Multi-Step Processing of Spatial Joins [COL] authors [VAL] T Brinkhoff, HP Kriegel\n[4]  "\
              "[COL] title [VAL] Scalable Sweeping-Based Spatial Join [COL] authors [VAL] L Arge, O Procopiuc"\
              "[COL] venue [VAL] VLDB, [COL] year [VAL] 1998.0\n[5]  [COL] title [VAL] Efficient Processing "\
              "of Spatial Joins Using R-trees, ACM SIGMOD Intl [COL] authors [VAL] T Brinkhoff, HP Kriegel,"\
              "B Seeger [COL] venue [VAL] Conference on Management of Data,\n[6]  [COL] title [VAL] Efficient"\
              "processing of spatial joins using R-trees. In: Peter Buneman, Sushil Jajodia eds [COL] authors "\
              "[VAL] T Brinkhoff, H Kriegel, B Seeger [COL] venue [VAL] 1993\n[7]  [COL] title [VAL] Performance"\
              "of Data-Parallel Spatial Operations [COL] authors [VAL] EG Hoel, H Samet [COL] venue [VAL] PROCEEDINGS"\
              "OF THE INTERNATIONAL CONFERENCE ON VERY LARGE  &hellip;, [COL] year [VAL] 1994.0\n[8]  [COL] title [VAL]"\
              "1993 . Efficient processing ofspatial joins using R-trees [COL] authors [VAL] T Brinkho, HP Kriegel, B Seeger"\
              "[COL] venue [VAL] Proc. ACM SIGMOD Int. Conf. on Management of\n[9]  [COL] title [VAL] tI. P. Kriegel and B."\
              "Seeger:&quot; Efficient Processing of Spatial Join Using R-trees [COL] authors [VAL] T Brinkhoff [COL] venue"\
              " [VAL] Proceedings of the 1990 ACM SIGMOD\n\n"\
              "Please respond in the correct format:\n{format_instructions}")
])

# Step 4: Format the prompt with parser's instructions
formatted_prompt = prompt.format_messages(format_instructions=parser.get_format_instructions())
human_prompt = formatted_prompt[0].content

print(human_prompt)

messages = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant trained in Entity Matching."),
    ("human", human_prompt)])

messages = [("system", "You are a helpful assistant trained in Entity Matching."),
            ("human", human_prompt)]

# formatted_prompt_2 = prompt.format_messages()
# messages = prompt.messages

# Step 5: Call the model
llm = ChatOpenAI(
    model_name="llama3.1:latest",
    # model_name = "llama3-70b-8192",
    # openai_api_base="http://localhost:9066/v1",
    openai_api_base="http://localhost:11434/v1",
    # openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key="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"
)
response = llm.invoke(messages)

# Step 6: Parse the response
try:
    result = parser.parse(response.content)
    print("Answer:", result.answer)
    print("Explanation:", result.explanation)
except Exception as e:
    print("Failed to parse structured output:")
    print(response.content)
