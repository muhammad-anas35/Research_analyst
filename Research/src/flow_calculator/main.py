from crewai import LLM, Agent, Crew, Process, Task # type: ignore
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource # type: ignore

# Create a knowledge source
content_source = CrewDoclingSource(
    file_paths=[
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination",
    ],
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(model="gemini/gemini-2.0-flash", temperature=0)

# Create an agent with the knowledge store
agent = Agent(
    role="About papers",
    goal="You know everything about the papers.",
    backstory="""You are a master at understanding papers and their content.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)
task = Task(
    description="Answer the following questions about the papers: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[
        content_source
    ],  # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
)
def main():
    result = crew.kickoff(
        inputs={
            "question": "Adversarial Attacks on LLMs"
        }
    )
    # print(result)

if __name__ == "__main__":
    main()

# # !/usr/bin/env python
# from crewai.flow.flow import Flow, listen, start  # type: ignore
# from dotenv import load_dotenv  # type: ignore
# from litellm import completion  # type: ignore
# import os 

# class ExampleFlow(Flow):
#     model = "gemini/gemini-2.0-flash"

#     @start()
#     def generate_city(self):
#         print("Starting flow")
#         # Each flow state automatically gets a unique ID
#         print(f"Flow State ID: {self.state['id']}")

#         response = completion(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": "Return the name of a random city in the world.",
#                 },
#             ],
#         )

#         random_city = response["choices"][0]["message"]["content"]
#         # Store the city in our state
#         self.state["city"] = random_city
#         print(f"Random City: {random_city}")

#         return random_city

#     @listen(generate_city)
#     def generate_fun_fact(self, random_city):
#         response = completion(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"Tell me a fun fact about {random_city}",
#                 },
#             ],
#         )

#         fun_fact = response["choices"][0]["message"]["content"]
#         # Store the fun fact in our state
#         self.state["fun_fact"] = fun_fact
#         return fun_fact



# flow = ExampleFlow()
# result = flow.kickoff()

# print(f"Generated fun fact: {result}")

 #!/usr/bin/env python
# from random import randint
# from pydantic import BaseModel
# from crewai.flow.flow import Flow, listen, start # type: ignore

# from flow_calculator.crews.poem_crew.poem_crew import PoemCrew


# class PoemState(BaseModel):
#     sentence_count: int = 1
#     poem: str = ""


# class PoemFlow(Flow[PoemState]):

#     @start()
#     def generate_sentence_count(self):
#         print("Generating sentence count")
#         self.state.sentence_count = randint(1, 5)

#     @listen(generate_sentence_count)
#     def generate_poem(self):
#         print("Generating poem")
#         result = (
#             PoemCrew()
#             .crew()
#             .kickoff(inputs={"sentence_count": self.state.sentence_count})
#         )

#         print("Poem generated", result.raw)
#         self.state.poem = result.raw

#     @listen(generate_poem)
#     def save_poem(self):
#         print("Saving poem")
#         with open("poem.txt", "w") as f:
#             f.write(self.state.poem)


# def kickoff():
#     poem_flow = PoemFlow()
#     poem_flow.kickoff()


# def plot():
#     poem_flow = PoemFlow()
#     poem_flow.plot()


# if __name__ == "__main__":
#     kickoff()
