from openai import OpenAI
from dataset_handler.oasst1 import getConversations

from endpoint.openai import sendToOpenAIEndpoint
from concurrent.futures import ThreadPoolExecutor

def main():
    print("Welcome to the Benchmark tool please select an option")
    
    parallel_conversations = 2
    message_count = 3
    min_context_messages = 1
    max_context_messages = 7

    #message_id, time_to_first_token, time_to_last_token, time_to_completion, total_tokens = sendToOpenAIEndpoint("test", "ollama", "http://188.172.229.26:3332/mistral/v1", [{"role": "user", "content": "Hello world"}], 512, True)
    
    #print("Time to first token: " + str(time_to_first_token))
    #print("Time to last token: " + str(time_to_last_token))
    #print("Time to completion: " + str(time_to_completion))
    #print("Total tokens: " + str(total_tokens))
    
    conversations = getConversations(message_count*parallel_conversations, min_context_messages, max_context_messages, "en", "train")
    print(len(conversations))
    
    def process_conversation(parallel_conversation, conversation):
        message_id, time_to_first_token, time_to_last_token, time_to_completion, total_tokens = sendToOpenAIEndpoint(str(parallel_conversation), "ollama", "https://llms.aiqrart.net/mistral/v1", conversations[conversation], "mistral", 512, True)
        

    with ThreadPoolExecutor() as executor:
        for parallel_conversation in range(parallel_conversations):
            for conversation in conversations: # doesn't work with rantge because its a dict. TODO change conversations to a list to make it work with range
                executor.submit(process_conversation, parallel_conversation, conversations[conversation+parallel_conversation]) # call with offset to avoid duplicate conversations
    
    

if __name__ == "__main__":
    main()