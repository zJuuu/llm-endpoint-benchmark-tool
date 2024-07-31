from openai import OpenAI
import time

def sendToOpenAIEndpoint(message_id, api_key, base_url, messages, model, max_tokens, stream=True):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    start_time = time.time()
    time_to_first_token = 9999999999
    time_to_last_token = 0
    total_tokens = 0
    
    try:
        for chat_completion in client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream
            ):
            #print(chat_completion.choices[0].delta.content)
            #total_tokens += 1 # Increment the total tokens because each chunk is a token (?)
            
            if time.time() < time_to_first_token:
                time_to_first_token = time.time()
            
            if time.time() > time_to_last_token:
                time_to_last_token = time.time()
            
        time_to_first_token = time_to_first_token - start_time
        time_to_last_token = time_to_last_token - start_time
        time_to_completion = time_to_last_token - time_to_first_token
    except Exception as e:
        print(e)
        print(message_id)
        #print(messages)
        
    return message_id, time_to_first_token, time_to_last_token, time_to_completion, total_tokens