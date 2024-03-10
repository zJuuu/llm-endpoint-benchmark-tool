from openai import OpenAI
from dataset_handler.oasst1 import getConversations
from endpoint.openai import sendToOpenAIEndpoint
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os

def main():
    print("Welcome to the Benchmark tool please select an option")
    
    parallel_conversations = 100
    message_count = 3
    min_context_messages = 1
    max_context_messages = 7
    
    # required variables
    OPENAI_ENDPOINT = os.getenv('OPENAI_ENDPOINT') if os.getenv('OPENAI_ENDPOINT') else "https://api.openai.com/v1"
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') if os.getenv('OPENAI_API_KEY') else "ollama"
    OPENAI_MAX_TOKENS = os.getenv('OPENAI_MAX_TOKENS') if os.getenv('OPENAI_MAX_TOKENS') else 512
    OPENAI_MODEL = os.getenv('OPENAI_MODEL') if os.getenv('OPENAI_MODEL') else "gpt-3.5-turbo"
    
    
    conversations = getConversations(message_count*parallel_conversations, min_context_messages, max_context_messages, "en", "train")
    
    results = dict()
    
    def process_conversation(parallel_conversation, conversations, conversation_keys):
        for conversation in conversations:
            
            conversation_key = conversation_keys[conversations.index(conversation)]
            message_id, time_to_first_token, time_to_last_token, time_to_completion, total_tokens = sendToOpenAIEndpoint(str(conversation_key), OPENAI_API_KEY, OPENAI_ENDPOINT+"/v1", conversation, OPENAI_MODEL, OPENAI_MAX_TOKENS, True)
            print("-------------------")
            print("Process: " + str(parallel_conversation))
            print("Message ID: " + message_id)
            print("Time to first token: " + str(time_to_first_token))
            print("Time to last token: " + str(time_to_last_token))
            print("Time to completion: " + str(time_to_completion))
            print("Total tokens: " + str(total_tokens))
            print("-------------------")
            
            results[message_id] = {
                "process": parallel_conversation,
                "time_to_first_token": time_to_first_token,
                "time_to_last_token": time_to_last_token,
                "time_to_completion": time_to_completion,
                "total_tokens": total_tokens
            }
            
        
        
    conversations_list = list(conversations.values())  # Convert dictionary values to a list
    conversations_keys = list(conversations.keys())  # Convert dictionary keys to a list
    conversationRange = int(len(conversations_list) / parallel_conversations)
    with ThreadPoolExecutor() as executor:
        for parallel_conversation in range(parallel_conversations):
            print("Starting process: " + str(parallel_conversation))
            start = parallel_conversation * conversationRange
            end = (parallel_conversation + 1) * conversationRange
            executor.submit(process_conversation, parallel_conversation, conversations_list[start:end], conversations_keys[start:end])
            
            
    print(results)
    # save as excel
    df = pd.DataFrame.from_dict(results, orient='index')
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_excel("./benchmark-results/" + timestamp + "_benchmark.xlsx")
            
    
    

if __name__ == "__main__":
    main()