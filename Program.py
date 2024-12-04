import streamlit as st
import openai
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
import os

llm = OpenAI(openai_api_key=st.secrets["OpenAI_Key"])


### Create the decision-making chain
issue_template = """You are an expert at booking airline tickets.
From the following review, determine whether the experience is one of the following three cases:
* Positive: The customer is not facing any issues and is having a good experience.
* NegativeFault: The customer is facing issues affecting their experience. The airline is responsible for the issue. For example, the airline lost the customer's luggage.
* NegativeNoFault: The customer is facing issues affecting their experience. However, the airline is not responsible for the issue. For example, a delay was caused by the weather.

Only respond with Positive, NegativeFault, or NegativeNoFault.

Review:
{review}

"""
issue_type_chain = (
    PromptTemplate.from_template(issue_template)
    | llm
    | StrOutputParser()
)


#### Case 1: Positive
positive_chain = PromptTemplate.from_template(
    """You are an experienced travel customer support agent.
    Thank the customer for their review and for choosing to fly with the airline.

Your response should follow these guidelines:
    1. Do not provide any reasoning behind the need for visa. Just respond professionally as a travel chat agent.
    2. Address the customer directly.
    3. You are answering as a bot. Don't leave your name or end with anything like "Best Regards".



Review:
{review}

"""
) | llm


#### Case 2: NegativeNoFault
nofault_chain = PromptTemplate.from_template(
    """You are an experienced travel customer support agent.
    Offer sympathies but explain to the customer that the airline is not liable in such situations.

Your response should follow these guidelines:
    1. Do not provide any reasoning behind the need for visa. Just respond professionally as a travel chat agent.
    2. Address the customer directly.
    3. You are answering as a bot. Don't leave your name or end with anything like "Best Regards".




Review:
{review}

"""
) | llm


#### Case 3: NegativeFault
fault_chain = PromptTemplate.from_template(
    """You are an experienced travel customer support agent.
    Display a message offering sympathies and inform the user that customer service will contact them soon to resolve the issue or provide compensation.

Your response should follow these guidelines:
    1. Do not provide any reasoning behind the need for visa. Just respond professionally as a travel chat agent.
    2. Address the customer directly.
    3. You are answering as a bot. Don't leave your name or end with anything like "Best Regards".




Review:
{review}

"""
) | llm



### Put all the chains together
branch = RunnableBranch(
    (lambda x: "Positive" in x["issue_type"], positive_chain),
    (lambda x: "NegativeNoFault" in x["issue_type"], nofault_chain),
    lambda x: fault_chain,
)
full_chain = {"issue_type": issue_type_chain, "review": lambda x: x["review"]} | branch

# streamlit app layout
st.title("Airline Experience Feedback")
prompt = st.text_input("Share with us your experience of the latest trip", "My trip was awesome")

# Run the chain
response = full_chain.invoke({"review": prompt})


st.write(
    response
)
