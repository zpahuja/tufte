objectives = openai.OpenAI()
messages = [{
    "role": "system",
    "content": 'Provide some goals for further observation via data science given a summary of a dataset. List them in order of revealing the most information to revealing the least information, with 10 ideas. Try not to repeat any ideas. ML applications can also be considered if they would reveal meaningful insights from the data'
    }, {"role": "system",
    "content": 'This is the summary: ' + summary['AI_Summarization']
    }]


summ = objectives.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    )

print(str(summ.choices[0].message.content))
