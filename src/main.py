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
    
    conversations = getConversations(message_count*parallel_conversations, min_context_messages, max_context_messages, "en", "train")
    
    def process_conversation(parallel_conversation, conversations, conversation_keys):
        #print(conversations)
        for conversation in conversations:
            conversation_key = conversation_keys[conversations.index(conversation)]
            message_id, time_to_first_token, time_to_last_token, time_to_completion, total_tokens = sendToOpenAIEndpoint(str(conversation_key), "ollama", "https://llms.aiqrart.net/mistral/v1", conversation, "mistral", 512, True)
            print("-------------------")
            print("Process: " + str(parallel_conversation))
            print("Message ID: " + message_id)
            print("Time to first token: " + str(time_to_first_token))
            print("Time to last token: " + str(time_to_last_token))
            print("Time to completion: " + str(time_to_completion))
            print("Total tokens: " + str(total_tokens))
            print("-------------------")
        
        
    conversations_list = list(conversations.values())  # Convert dictionary values to a list
    conversations_keys = list(conversations.keys())  # Convert dictionary keys to a list
    conversationRange = int(len(conversations_list) / parallel_conversations)
    with ThreadPoolExecutor() as executor:
        for parallel_conversation in range(parallel_conversations):
            print("Starting process: " + str(parallel_conversation))
            start = parallel_conversation * conversationRange
            end = (parallel_conversation + 1) * conversationRange
            executor.submit(process_conversation, parallel_conversation, conversations_list[start:end], conversations_keys[start:end])
            
    
    

if __name__ == "__main__":
    main()