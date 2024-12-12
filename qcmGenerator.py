from openai import OpenAI

def get_questions(course, api_key="sk-proj-KEY", model="gpt-4o-mini"):
        
        client = OpenAI(api_key=api_key)

        system_prompt= """
            <CONTEXT>You are a teacher. You will be given a course either in English or French. Your job is to create a quick test from this course and only this course. Write 2 to 5 questions about the course, in the language that the course is written in.</CONTEXT>
            <FORMAT> You are expected to return the questions in a json format, as follows (this is an example): 
            
            [
          {
            "question": 'Which book did Marguerite Duras write ?',
            "options": ['Hamlet', 'L'amant', 'Bonjour tristesse', 'L'espèce humaine'],
            "answer": 'L'amant'
          },
          {
            "question": "Who wrote 'Les Misérables' ?",
            "options": ['Victor Hugo', 'Emile Zola', 'Marcel Proust', 'Albert Camus'],
            "answer": 'Victor Hugo'
          }
        ]
            
            DO NOT ADD ANYTHING ELSE, NO ``json `` OR THINGS LIKE THAT. JUST THE JSON.</FORMAT>
    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": course}
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature = 0.05
        )

        message_content = chat_completion.choices[0].message.content
        return(message_content)