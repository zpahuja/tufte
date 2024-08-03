class Visgenerator():
  def __init__(self) -> None:
    self.client = OpenAI()

  def visualize_data(self, code):
    try:
      exec(str(code))
    except SyntaxError:
      code = self.fix_Code(code)
    return(exec(str(code)))

  def generate_code(self, goals, dataset_name):
    self.tools = [PythonREPLTool()]
    template = '''

    Make python code to visualize the provided goal. If you have a Final Answer, do not try to use any The name of the dataset is + ''' + dataset_name + '''. You have access to the following tools:

    {tools}

    Use the following format:

    Question:
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the complete final code to the goal, not one section of it. This code must compile properly.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    '''

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(self.client, self.tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    code_generation = agent_executor.invoke({"input": {goals}})
    self.visualize_data(code_generation['output'])
  
  def fix_Code(self, code):
    template = '''

    You are given python code which when complied gives a syntax error. You must try to fix the code so it will compile properly. You have access to the following tools:

    {tools}

    Use the following format:

    Question:
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the same code you are given, except without any syntax errors. This code must be able compile properly.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    '''
    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(self.client, self.tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    code_generation = agent_executor.invoke({"input": {code}})

    self.visualize_data(code)

  def visualize(self, goal, dataset_name):
    code = self.generate_code(goal, dataset_name)
    graph = self.visualize_data(code)

    return(graph)

    






