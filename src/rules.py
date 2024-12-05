from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from model.boid import Rule, BoidType

from src.config import config


def generate_rules(desires: str):
    system_prompt_template = """
    You are an AI agent specialized in the BOID architecture. Your task is to convert a given conference schedule and the user's
    obligations and desires into BOID rules in the format:
    ```predicate -type-> conclusion```
    
    Where:
    - **predicate** represents a condition.
    - **type** specifies the mental attitude (Belief: B, Obligation: O, Intention: I, Desire D).
    - **conclusion** is the result
    
    Follow these guidelines:
    1. Use the schedule entries (timeslot id and presentation name) to derive relevant predicates.
    2. Map the user's obligations and desires into corresponding BOID components.
    3. Ensure outputs are consistent with the format.
    4. Provide output as a list.
    5. Timeslots should be formatted as `timeslot_#id#_#presentation_id#`.
    6. For each presentation there should exist a rule `true -T-> presentation_#id#`, where T is a mental attitude type.
    7. Presentations should be linked to timeslots as `presentation_#id# -B-> timeslot_#id#_#presentation_id#`.
    
    Below is an example:
    ## Input:
    ### Schedule
    {
        "schedule": [
            {
              "date": "2024-11-30",
              "timeslots": [
                {
                  "id": 1,
                  "presentations": [
                    {
                      "id": 6,
                      "name": "Prof. Brian Lee",
                      "topic": "Advanced Machine Learning"
                    },
                    {
                      "id": 5,
                      "name": "Dr. Alice Johnson",
                      "topic": "Introduction to AI"
                    }
                  ]
                },
                {
                  "id": 2,
                  "presentations": [
                    {
                      "id": 8,
                      "name": "Mr. David Green",
                      "topic": "Blockchain for Beginners"
                    },
                    {
                      "id": 7,
                      "name": "Dr. Catherine Miller",
                      "topic": "Quantum Computing Basics"
                    }
                  ]
                }
              ]
            }
        }
    ### Obligations
    Machine Learning
    
    ### Desires
    Advanced Machine Learning, Introduction to AI, Quantum Computing Basics
    
    ### Output 
    presentation_6 -B-> timeslot_1_6
    presentation_5 -B-> timeslot_1_5
    presentation_8 -B-> timeslot_2_8
    presentation_7 -B-> timeslot_2_7
    true -O-> presentation_6
    true -D-> presentation_5
    true -D-> presentation_8
    true -D-> presentation_7  
    
    Now do the same for the following. 
    Remember to only reply with the BOID rules!
    """

    system_prompt = PromptTemplate(system_prompt_template).format()

    user_template = """
    ### Schedule
    {schedule}
    
    ### Obligations
    {obligations}
    
    ### Desires
    {desires}
    """

    user_prompt_template = PromptTemplate(user_template)
    user_prompt = user_prompt_template.format(schedule=config.schedule.to_json(), obligations=config.user.obligations, desires=desires)
    config.llm.temperature = 0.2
    response = config.llm.chat(messages=[
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt)
    ])

    return response.message.content

def encode_rules(rules: str)-> list[Rule]:
    proto_rules = []

    for rule in rules.split("\n"):
        rule_type = rule.split("-")[1].split("->")[0]

        rule_boid_type = BoidType.BELIEF

        if rule_type == "B":
            rule_boid_type = BoidType.BELIEF
        elif rule_type == "O":
            rule_boid_type = BoidType.OBLIGATION
        elif rule_type == "I":
            rule_boid_type = BoidType.INTENTION
        elif rule_type == "D":
            rule_boid_type = BoidType.DESIRE



        proto_rules.append(Rule(head=rule.split("-")[0], complement=rule.split(">")[-1], rule_type=rule_boid_type))

    return proto_rules
