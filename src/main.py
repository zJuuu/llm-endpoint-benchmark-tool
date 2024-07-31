from openai import OpenAI
from dataset_handler.oasst1 import getConversations
from dataset_handler.lmsyschat1m import getLMSYSConversations
from endpoint.openai import sendToOpenAIEndpoint
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

def main():
    print("Welcome to the Benchmark tool please select an option")
    
    PARALLEL_CONVERSATIONS = int(os.getenv('PARALLEL_CONVERSATIONS')) if os.getenv('PARALLEL_CONVERSATIONS') else 1
    MESSAGE_COUNT = int(os.getenv('MESSAGE_COUNT')) if os.getenv('MESSAGE_COUNT') else 3
    MIN_CONTEXT_MESSAGES = int(os.getenv('MIN_CONTEXT_MESSAGES')) if os.getenv('MIN_CONTEXT_MESSAGES') else 3
    MAX_CONTEXT_MESSAGES = int(os.getenv('MAX_CONTEXT_MESSAGES')) if os.getenv('MAX_CONTEXT_MESSAGES') else 7
    
    # required variables
    OPENAI_ENDPOINT = os.getenv('OPENAI_ENDPOINT') if os.getenv('OPENAI_ENDPOINT') else "https://api.openai.com"
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') if os.getenv('OPENAI_API_KEY') else "ollama"
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS')) if os.getenv('OPENAI_MAX_TOKENS') else 512
    OPENAI_MODEL = os.getenv('OPENAI_MODEL') if os.getenv('OPENAI_MODEL') else "gpt-3.5-turbo"
    
    LANGUAGE = os.getenv('LANGUAGE') if os.getenv('LANGUAGE') else "en"
    DATASET = os.getenv('DATASET') if os.getenv('DATASET') else "lmsyschat1m"
    
    print("Starting benchmark with the following parameters:")
    print("Parallel conversations: " + str(PARALLEL_CONVERSATIONS))
    print("Message count: " + str(MESSAGE_COUNT))
    print("Min context messages: " + str(MIN_CONTEXT_MESSAGES))
    print("Max context messages: " + str(MAX_CONTEXT_MESSAGES))
    print("OpenAI endpoint: " + OPENAI_ENDPOINT)
    print("OpenAI API key: " + OPENAI_API_KEY)
    print("OpenAI max tokens: " + str(OPENAI_MAX_TOKENS))
    print("OpenAI model: " + OPENAI_MODEL)
    print("Language: " + LANGUAGE)
    
    
    if DATASET == "oasst1":
        conversations = getConversations(MESSAGE_COUNT*PARALLEL_CONVERSATIONS, MIN_CONTEXT_MESSAGES, MAX_CONTEXT_MESSAGES, LANGUAGE, "train")
    elif DATASET == "lmsyschat1m":
        conversations = getLMSYSConversations(MESSAGE_COUNT*PARALLEL_CONVERSATIONS, MIN_CONTEXT_MESSAGES, MAX_CONTEXT_MESSAGES, LANGUAGE)
    
    results = dict()
    
    def process_conversation(parallel_conversation, conversations, conversation_keys):
        for conversation in conversations:
            
            conversation_key = conversation_keys[conversations.index(conversation)]
            message_id, time_to_first_token, time_to_last_token, time_to_completion, total_tokens = sendToOpenAIEndpoint(str(conversation_key), str(OPENAI_API_KEY), str(OPENAI_ENDPOINT)+"/v1", conversation, str(OPENAI_MODEL), int(OPENAI_MAX_TOKENS), True)
            print("-------------------")
            print("Process: " + str(parallel_conversation))
            print("Message ID: " + message_id)
            print("Time to first token: " + str(time_to_first_token))
            print("Time to last token: " + str(time_to_last_token))
            print("Time to completion: " + str(time_to_completion))
            #print("Total tokens: " + str(total_tokens)) # not sure if this is calculated correctly
            print("-------------------")
            
            results[message_id] = {
                "process": parallel_conversation,
                "time_to_first_token": time_to_first_token,
                "time_to_last_token": time_to_last_token,
                "time_to_completion": time_to_completion,
                #"total_tokens": total_tokens #not sure if this is calculated correctly
            }
            
        
        
    conversations_list = list(conversations.values())  # Convert dictionary values to a list
    conversations_keys = list(conversations.keys())  # Convert dictionary keys to a list
    conversationRange = int(len(conversations_list) / PARALLEL_CONVERSATIONS)
    with ThreadPoolExecutor() as executor:
        for parallel_conversation in range(PARALLEL_CONVERSATIONS):
            print("Starting process: " + str(parallel_conversation))
            start = parallel_conversation * conversationRange
            end = (parallel_conversation + 1) * conversationRange
            executor.submit(process_conversation, parallel_conversation, conversations_list[start:end], conversations_keys[start:end])
            
            
    # save as excel
    df = pd.DataFrame.from_dict(results, orient='index')
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_excel("./benchmark-results/" + timestamp + "_benchmark.xlsx")
            
    
    

if __name__ == "__main__":
    main()