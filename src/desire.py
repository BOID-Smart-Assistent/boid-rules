from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from numpy.distutils.system_info import system_info

from src.config import config


def get_desire():
    if config.online_mode:
        return get_desire_online()
    else:
        return get_desire_offline()


def get_desire_online() -> str:
    pass

def get_desire_offline() -> str:
    system_template = """
        You are an AI assistant. Your task is to extract topics from a provided schedule that are of interest to the user. 
        The user's interests may not perfectly match the topic names in the schedule, so you must infer connections based on context, synonyms, and related concepts. 
        Provide the topics of interest in a comma-separated list.
        
        Here's an example:
        
        ### User Interests
        Ethics, Natural Language Processing, Quantum Computing 
        
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
        
        ### Output
        Advanced Machine Learning, Introduction to AI, Quantum Computing Basics 
        
        Now, do the same for the following. Remember to only reply with the topics in a comma-separated list.
    """

    system_prompt_template = PromptTemplate(system_template)
    system_prompt = system_prompt_template.format()

    user_template = """
        ### User Interests
        {interests}
    
        ### Schedule
        {schedule} 
        
        ### Output 
        Extracted topics (comma-separate)
    """

    user_prompt_template = PromptTemplate(user_template)
    user_prompt = user_prompt_template.format(interests=','.join(config.user.interests), schedule=config.schedule.to_json())
    #
    # messages = [
    #     {"role": "system", "content": "You are an assistant that convert only like topics into desired topics."},
    #     {"role": "user", "content": f"Convert following input --> like: quantum, ethics ; dislike: robotics, education \n reference: {config.schedule.to_json()} become this format --> DESIRES: <insert title>. Output only the requested format and exact title from reference that match with 'like' keywords."},
    #     {"role": "assistant", "content": "DESIRES: Quantum Computing Basics,Quantum Cryptography, Quantum Machine Learning, Quantum Algorithms, Data Ethics in AI, AI and Society"},
    #     # {"role": "user", "content": f"Now convert these keywords {input('like and dislike keywords:')} into desires."},
    # ]

    # if config.debug:
    #     messages.append(ChatMessage(role=MessageRole.USER, content=f"Now convert these keywords {input('like and dislike keywords:')} into desires."))
    # else:
    #     messages.append(ChatMessage(role=MessageRole.USER, content=f"Now convert these keywords {','.join(config.user.interests)} into desires."))

    config.llm.temperature = 0.2
    response = config.llm.chat(messages=[ChatMessage(role=MessageRole.SYSTEM, content=system_prompt), ChatMessage(role=MessageRole.USER, content=user_prompt)])

    return response.message.content
