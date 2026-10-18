import os
from dotenv import load_dotenv
import anthropic
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pprint import pprint

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

@tool
def get_weather(location:str)->str:
  """Use this to get the weather details for a location"""
  return f"The weather of {location} is Sunny"


model = init_chat_model("claude-sonnet-5")

#region Running a tool loop by ourself. So basically here we will send the Human message, call the tool and then send the tool reply to the model again by ourselves, to get the response
model_with_tool = model.bind_tools([get_weather])

# Step1: Model generates tool calls
messages = [{"role":"user", "content": "What is the weather like in Boston?"}]
ai_msg = model_with_tool.invoke(messages)
messages.append(ai_msg)

# Step2 : Execute tools and collect results
for tool_call in ai_msg.tool_calls:
  tool_result = get_weather.invoke(tool_call)
  messages.append(tool_result)

# Step3 : Pass results back to model for final response
final_response = model_with_tool.invoke(messages)
print(final_response.content)

#endregion


#region calling a tool through an agent. 

weather_agent = create_agent(model=model,
                             tools=[get_weather]
                             )

agent_result = weather_agent.invoke({"messages":messages})
print("-- agent response--")
for message in agent_result["messages"]:
    message.pretty_print()