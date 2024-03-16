
import os
from datasets import load_dataset


def getConversations(conversationCount: int, minContextMessages: int, maxContextMessages: int, language: str, split: str = "train") -> 'dict[str, list[dict]]':
    hf_dataset_name = "OpenAssistant/oasst1"
    dataset = load_dataset(hf_dataset_name, split=split)
    
    messageTreesRaw = dict()
    for row in dataset:
        if row["message_tree_id"] not in messageTreesRaw:
            messageTreesRaw[row["message_tree_id"]] = [row]
        else:
            messageTreesRaw[row["message_tree_id"]].append(row)
            
    # remove all message trees in specific language if set
    tempMessageTreesRaw = messageTreesRaw.copy() # create copy because we are deleting from the original directly
    if language:
        for messageTree in tempMessageTreesRaw:
            for message in tempMessageTreesRaw[messageTree]:
                if message["lang"] != language:
                    if messageTreesRaw.get(messageTree) is not None:
                        del messageTreesRaw[messageTree]
    
    # remove unused columns and rename text to content
    tempMessageTreesRaw = messageTreesRaw.copy()
    for messageTree in tempMessageTreesRaw:
        messageTreesRaw[messageTree] = []
        for message in tempMessageTreesRaw[messageTree]:
            tempMessage = dict()
            tempMessage["parent_id"] = message["parent_id"]
            tempMessage["message_id"] = message["message_id"]
            tempMessage["role"] = "assistant" if message["role"] == "assistant" else "user"
            tempMessage["content"] = message["text"]
            messageTreesRaw[messageTree].append(tempMessage)
            #print(message["parent_id"], message["message_id"], message["role"], message["text"])
        
        #print("----")
                
    # make sure messages are in correct order and remove ids from the messages
    tempMessageTreesRaw = messageTreesRaw.copy()
    for messageTree in tempMessageTreesRaw:
        messageTreesRaw[messageTree] = recursiveMessageSort(tempMessageTreesRaw[messageTree], None)
        for message in messageTreesRaw[messageTree]:
            del message["message_id"]
            del message["parent_id"]
    
    if os.getenv('IS_INSTRUCT_MODEL') == 'True':
    # merge messages where role is the same as the previous message
        tempMessageTreesRaw = messageTreesRaw.copy()
        for messageTree in tempMessageTreesRaw:
            tempMessageTreesRaw[messageTree] = messageTreesRaw[messageTree].copy()
            messageTreesRaw[messageTree] = []
            lastRole = None
            lastMessage = None
            for message in tempMessageTreesRaw[messageTree]:
                if lastRole == message["role"]:
                    lastMessage["content"] += " " + message["content"]
                else:
                    messageTreesRaw[messageTree].append(message)
                    lastMessage = message
                lastRole = message["role"]
    
            
    if minContextMessages and maxContextMessages:
        tempMessageTreesRaw = messageTreesRaw.copy()
        for messageTree in tempMessageTreesRaw:
            if len(tempMessageTreesRaw[messageTree]) < minContextMessages or len(tempMessageTreesRaw[messageTree]) > maxContextMessages:
                del messageTreesRaw[messageTree]
            
    messageTrees = dict()     
    if conversationCount:
        counter = 0
        for messageTree in messageTreesRaw:
            counter += 1
            if counter > conversationCount:
                break
            messageTrees[messageTree] = messageTreesRaw[messageTree]
    else:
        messageTrees = messageTreesRaw
        
    #messageList = []
    #for messageTree in messageTrees:
    #    messageList.append(messageTrees[messageTree])
        
    return messageTrees

def recursiveMessageSort(messageTree: 'list[dict]', parent_id: str) -> 'list[dict]':
    tempMessageTree = messageTree.copy()
    for message in tempMessageTree:
        if message["parent_id"] == parent_id:
            messageTree.remove(message)
            messageTree.append(message)
            recursiveMessageSort(messageTree, message["message_id"])
    return messageTree