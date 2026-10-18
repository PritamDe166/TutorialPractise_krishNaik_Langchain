## langchain Version V1
import os
from dotenv import load_dotenv
import anthropic
from langchain.agents import create_agent

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# region get list of models by calling anthropic get api
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

for model in client.models.list():
  print(f"Models available: {model.id} - {model.display_name}")

# endregion


# region create a custom tool
def get_weather(city:str) -> str:
  """Get the weather for a city."""
  return f"The weather in {city} is sunny"
# endregion

# region Create agents using langchain
agent = create_agent(
  model="claude-sonnet-4-6",
  tools=[get_weather],
  system_prompt="You are a helpful assistant"
)
# endregion


response = agent.invoke({"messages":"What is the weather in Bangalore"})
for m in response["messages"]:
  m.pretty_print()

