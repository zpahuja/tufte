class goalExplorer():
  def getGoals(self, summary: dict, n_goals : int = 5):
    objectives = openai.OpenAI()
    messages = [{
      "role": "system",
     "content": 'Provide some goals for further observation via data science given a summary of a dataset. List them in order of revealing the most information to revealing the least information, with only ' + str(n_goals) + ' ideas. Try not to repeat any ideas. ML applications can also be considered if they would reveal meaningful insights from the data. Do not give broad goals like "Do a complete analysis of the dataset"'
     }, {"role": "system",
     "content": 'This is the summary: ' + str(summary)
     }]


    summ = objectives.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    )

    answer = str(summ.choices[0].message.content)

    return(answer)

goals_client = goalExplorer()

goal = goals_client.getGoals(summary)

print(goal)
