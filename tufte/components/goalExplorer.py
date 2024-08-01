class goalExplorer():
  def getGoals(self, summary: dict, n_goals : int = 5):
    objectives = openai.OpenAI()
    messages = [{"role": "system",
    "content": 'Not only provide a goal/hypothesis/question that we are trying to answer given this dataset, but also the visualization & statistics to be computed. You should produce the reasoning first, and ONLY THEN move on to the goals and visualizations. Make sure to generate the visualizations and statistics for each goal'
    },
     {"role": "system",
    "content": 'Provide some goals for further observation via data science given a summary of a dataset. Ouput it in a JSON schema. List them in order of revealing the most information to revealing the least information and label each idea with how important it is. Try not to repeat any ideas. ML applications can also be considered if they would reveal meaningful insights from the data. Only provide ' + str(n_goals) + ' goals. This is the summary: ' + str(summary)
    }]



    summ = objectives.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    response_format = {"type" : 'json_object'}
    )

    answer = str(summ.choices[0].message.content)

    return(answer)

goals_client = goalExplorer()

goal = goals_client.getGoals(summary)

print(goal)
