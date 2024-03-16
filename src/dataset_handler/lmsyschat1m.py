
import os
from datasets import load_dataset


def getLMSYSConversations(conversationCount: int, minContextMessages: int, maxContextMessages: int, language: str) -> 'dict[str, list[dict]]':
    hf_dataset_name = "lmsys/lmsys-chat-1m"
    dataset = load_dataset(hf_dataset_name, split='train[:5%]')
    
    messageTreesRaw = dict()
    for row in dataset:
        if row["language"] != language:
            continue
        messageTreesRaw[row["conversation_id"]] = row["conversation"]
    
    
    # if last message of conversation is from "assistant" remove the message from the conversation
    tempMessageTreesRaw = messageTreesRaw.copy()
    for messageTree in tempMessageTreesRaw:
        if messageTreesRaw[messageTree][-1]["role"] == "assistant":
            messageTreesRaw[messageTree].pop()
    
    # remove messages where the conversation is too short or too long
    tempMessageTreesRaw = messageTreesRaw.copy()
    for messageTree in tempMessageTreesRaw:
        if len(tempMessageTreesRaw[messageTree]) < minContextMessages or len(tempMessageTreesRaw[messageTree]) > maxContextMessages:
            del messageTreesRaw[messageTree]
            
    # remove conversations if conversationCount is set
    if conversationCount:
        counter = 0
        tempMessageTreesRaw = messageTreesRaw.copy()
        for messageTree in tempMessageTreesRaw:
            counter += 1
            if counter > conversationCount:
                del messageTreesRaw[messageTree]
                
    
    return messageTreesRaw