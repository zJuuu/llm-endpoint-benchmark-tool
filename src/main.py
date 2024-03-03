import sys
from openai import OpenAI
from dataset_handler.oasst1 import getConversations

from endpoint.openai import sendToOpenAIEndpoint

def main():
    print("Welcome to the Benchmark tool please select an option")

    #message_id, time_to_first_token, time_to_last_token, time_to_completion, total_tokens = sendToOpenAIEndpoint("test", "ollama", "http://188.172.229.26:3332/mistral/v1", [{"role": "user", "content": "Hello world"}], 512, True)
    
    #print("Time to first token: " + str(time_to_first_token))
    #print("Time to last token: " + str(time_to_last_token))
    #print("Time to completion: " + str(time_to_completion))
    #print("Total tokens: " + str(total_tokens))
    
    conversations = getConversations(1, 1, 7, "en", "train")
    print(len(conversations))
    

if __name__ == "__main__":
    main()